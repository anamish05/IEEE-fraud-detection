import torch
from torch.utils.data import Dataset

class FraudMultiModalDataset(Dataset):
    def __init__(self, x_cont, x_cat, x_seq, seq_mask, x_text, y=None):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_cat  = torch.tensor(x_cat, dtype=torch.int64)
        self.x_seq  = torch.tensor(x_seq, dtype=torch.float32)
        self.mask   = torch.tensor(seq_mask, dtype=torch.bool)
        self.x_text = torch.tensor(x_text, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        item = (self.x_cont[idx], self.x_cat[idx], self.x_seq[idx], self.mask[idx], self.x_text[idx])
        if self.y is not None:
            return (*item, self.y[idx])
        return item