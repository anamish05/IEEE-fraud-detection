import torch
import torch.nn as nn
from src.models.modules.transformers import TransactionSequenceTransformer




class FraudMultiModalModel(nn.Module):
    def __init__(self, emb_szs: list[tuple[int, int]], n_cont: int, d_seq: int, d_text:int=384, d_cat_proj: int = 32, cat_dropout:float=0.1):
        super().__init__()
        
        # 1. Static Branch (MLP)
        self.num_cats=len(emb_szs)
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_szs])

        # Linear layer per embedding to project variable sizes to uniform d_cat_proj
        self.emb_projections = nn.ModuleList([
            nn.Linear(d, d_cat_proj) for _, d in emb_szs
        ])

        # nn.Dropout1d treats dimension 1 as channels:
        # Input: (B, N_cat, d_cat_proj) -> randomly drops entire N_cat channels
        self.cat_spatial_dropout = nn.Dropout1d(p=cat_dropout)
        
        total_cat_dim = self.num_cats * d_cat_proj

        self.bn_cont = nn.BatchNorm1d(n_cont) # keeps the continuous features at the same numerical magnitude as the embeddings
        self.static_mlp = nn.Sequential(
            nn.Linear(total_cat_dim + n_cont, 256),
            nn.BatchNorm1d(256), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # 2. Sequence Branch (Transformer)
        self.seq_transformer = TransactionSequenceTransformer(
            d_in=d_seq, d_model=64, nhead=4, num_layers=2
        )
        
        # 3. Device-Text Branch
        self.text_head = nn.Sequential(
            nn.Linear(d_text, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 4. Fusion Classifier Head
        # Concatenated sizes: 128 (static) + 64 (seq) + 32 (text) = 224
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64 + 32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output single raw logit
        )

    def forward(self, x_cont, x_cat, x_seq, seq_mask, x_text):
        # Branch 1: Static

        # 1. Project each category to (B, d_cat_proj)
        projected_embs = [
            proj(emb(x_cat[:, i]))
            for i, (emb, proj) in enumerate(zip(self.embeddings, self.emb_projections))
        ]
        
        # 2. Stack into 3D: (B, N_cat, d_cat_proj)
        stacked_cat = torch.stack(projected_embs, dim=1)
        
        # 3. Apply Dropout1d across the category dimension
        # Entire category slots are zeroed out with probability cat_dropout_p
        dropped_cat = self.cat_spatial_dropout(stacked_cat)
        
        # 4. Flatten back to 2D: (B, N_cat * d_cat_proj)
        x_emb = dropped_cat.flatten(start_dim=1)
    
        h_static = self.static_mlp(torch.cat([x_emb, self.bn_cont(x_cont)], dim=1))  # (b, total_emb_dim+n_cont) -> (b, 128)
        
        # Branch 2: Sequence
        h_seq = self.seq_transformer(x_seq, seq_mask) # (b, 64)
        
        # Branch 3: Text
        h_text = self.text_head(x_text) # (b, 32)
        
        # Fusion
        fused = torch.cat([h_static, h_seq, h_text], dim=1)
        return self.classifier(fused).squeeze(-1)