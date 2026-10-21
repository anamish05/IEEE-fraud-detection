import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List


class FraudDLRMDataset(Dataset):
    """Lightweight dataset for static dense + categorical DLRM inputs."""
    def __init__(self, X_cont, X_cat, y=None):
        self.X_cont = torch.as_tensor(X_cont, dtype=torch.float32)
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        return self.X_cont[idx], self.X_cat[idx]


class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model (DLRM) adapted for tabular fraud detection.
    All categorical embeddings and the Bottom MLP share dimension `d`.
    """
    def __init__(
        self,
        n_dense: int,
        cardinalities: List[int],
        d: int = 16,
        bottom_mlp_dims: List[int] = [256, 128],
        top_mlp_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        emb_dropout: float = 0.1
    ):
        super().__init__()
        self.d = d
        self.num_cat = len(cardinalities)

        # 1. Shared-dimension Embedding tables (0 = Missing, 1 = Rare)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=d)
            for card in cardinalities
        ])
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 2. Bottom MLP: transforms n_dense -> d
        bottom_layers = [nn.BatchNorm1d(n_dense)]
        in_dim = n_dense
        for h_dim in bottom_mlp_dims:
            bottom_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        bottom_layers.extend([
            nn.Linear(in_dim, d),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
            ])
        self.bottom_mlp = nn.Sequential(*bottom_layers)

        # 3. Pairwise Dot-Product Interaction Indices
        # Total vectors = 1 (dense bottom MLP) + F (categorical embeddings)
        total_vecs = self.num_cat + 1
        tril_indices = torch.tril_indices(row=total_vecs, col=total_vecs, offset=-1)
        self.register_buffer("tril_row", tril_indices[0])
        self.register_buffer("tril_col", tril_indices[1])

        num_interactions = (total_vecs * (total_vecs - 1)) // 2

        # 4. Top MLP: takes [bottom_mlp_out, flat_dot_products] -> 1 logit
        top_in_dim = d + num_interactions
        top_layers = [nn.BatchNorm1d(top_in_dim)]
        in_dim = top_in_dim
        for h_dim in top_mlp_dims:
            top_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        top_layers.append(nn.Linear(in_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Dense representation: (B, d)
        dense_emb = self.bottom_mlp(x_cont)

        # Categorical representations: list of (B, d) -> stacked to (B, F, d)
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_stack = torch.stack(cat_embs, dim=1)
        cat_stack = self.emb_dropout(cat_stack)

        # Combine dense and sparse vectors: (B, F + 1, d)
        all_vecs = torch.cat([dense_emb.unsqueeze(1), cat_stack], dim=1)

        # L2-normalize vectors before dot product to prevent gradient spikes
        all_vecs_norm = nn.functional.normalize(all_vecs, p=2, dim=-1)

        # Pairwise dot products via batch matrix multiplication: (B, F+1, F+1)
        interactions_mat = torch.bmm(all_vecs_norm, all_vecs_norm.transpose(1, 2))

        # Extract strict lower triangle: (B, (F+1)*F // 2)
        interactions_flat = interactions_mat[:, self.tril_row, self.tril_col]

        # Concatenate original dense representation + interactions -> Top MLP
        top_input = torch.cat([dense_emb, interactions_flat], dim=1)
        logits = self.top_mlp(top_input).squeeze(-1)
        return logits