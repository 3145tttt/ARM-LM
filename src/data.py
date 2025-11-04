import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def create_train_val_split(data_path, split_ratio=0.99):
    train_path = f"./data/train_{int(split_ratio * 100)}.txt"
    val_path = f"./data/validation_{int(split_ratio * 100)}.txt"

    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path

    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print(f"Train size = {len(train_lines)}")
    print(f"Validational size = {len(val_lines)}")
    with open(train_path, 'w') as f:
        f.writelines(train_lines)
    
    with open(val_path, 'w') as f:
        f.writelines(val_lines)
    
    return train_path, val_path


class CorpusDataset(Dataset):

    def __init__(self, tokenizer_json, data_path, context_length):
        self.context_length = context_length
        tokenizer = Tokenizer.from_file(tokenizer_json)

        with open(data_path, 'r') as f:
            text = f.read()
        
        tokenized_output = tokenizer.encode(text)
        self.data = torch.tensor(tokenized_output.ids, dtype=torch.long)
        print(f"Corpus has {len(self.data):,} tokens from file {data_path}.")

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.context_length + 1]
        return chunk[:-1]
    