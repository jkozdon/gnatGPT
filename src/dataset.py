import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharTokenizer

class CharDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharTokenizer, context_len: int):
        self.context_len = context_len
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        # each index i yields a window starting at i
        # last valid start: len - context_len - 1
        return len(self.ids) - self.context_len

    def __getitem__(self, idx):
        chunk = self.ids[idx: idx + self.context_len + 1]   # length context_len+1
        x = chunk[:-1] # input: tokens 0..T-1
        y = chunk[1:]  # input: tokens 1..T
        return (x, y)

def make_dataloader(text, tokenizer, context_len, batch_size, shuffle=True):
    dataset = CharDataset(text, tokenizer, context_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
