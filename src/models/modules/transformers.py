import torch
import torch.nn as nn

class TransactionSequenceTransformer(nn.Module):
    def __init__(
        self,
        d_in: int = 5,           # Number of sequential features per step (D_seq)
        d_model: int = 64,       # Internal representation size
        nhead: int = 4,          # Attention heads (d_model must be divisible by nhead)
        num_layers: int = 2,     # Transformer encoder depth
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_len: int = 15
    ):
        super().__init__()
        
        # 1. Project raw features into d_model dimensional space
        self.input_proj = nn.Linear(d_in, d_model)
        
        # 2. Learnable positional embeddings for chronological step positions
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # 3. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Inputs are (Batch, Seq_len, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 4. Dimension reduction head for fusion
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x_seq : torch.FloatTensor, shape (Batch, max_len, d_in)
        padding_mask : torch.BoolTensor, shape (Batch, max_len)
                       True indicates padded/dummy steps to mask out.
                       
        Returns:
        --------
        h_seq : torch.FloatTensor, shape (Batch, d_model)
        """
        B, L, _ = x_seq.shape
        
        # 1. Project input features
        h = self.input_proj(x_seq)  # (B, L, d_model)
        
        # 2. Add positional encoding
        h = h + self.pos_embedding[:, :L, :]
        h = self.dropout(h)
        
        # 3. Transformer Self-Attention
        # PyTorch requires src_key_padding_mask: True = ignore, False = attend
        # If an entire sequence is padded (edge case), unmask the last token to prevent NaN gradients
        all_padded = padding_mask.all(dim=-1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, -1] = False
            
        h_trans = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        h_trans = self.layer_norm(h_trans)
        
        # 4. Pooling: Extract the representation of the current transaction (last token, index -1)
        # Because we aligned right-to-left, index -1 is always the target transaction t
        h_current = h_trans[:, -1, :]  # (B, d_model)
        
        # 5. Optional: Masked Mean Pooling across past context
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # 1 for valid, 0 for pad
        h_mean = (h_trans * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-5)
        
        # Combine current state with aggregate sequence trajectory
        h_seq = self.out_proj(h_current + h_mean)
        
        return h_seq