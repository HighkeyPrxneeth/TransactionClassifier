import torch
from torch.utils.data import Dataset
import numpy as np
import os

class TransactionCategoryDataset(Dataset):
    def __init__(self, data_dir='.', mmap_mode=None):
        """
        Args:
            data_dir: Directory containing .npy files.
            mmap_mode: 'r' for read-only memory mapping (saves RAM for huge datasets).
        """
        self.txn_embeds = np.load(os.path.join(data_dir, 'txn_embeds.npy'), mmap_mode=mmap_mode)
        self.cat_embeds_flat = np.load(os.path.join(data_dir, 'cat_embeds_flat.npy'), mmap_mode=mmap_mode)
        self.cat_offsets = np.load(os.path.join(data_dir, 'cat_offsets.npy')) # Offsets are usually small, no mmap needed
        
        self.length = len(self.txn_embeds)
        print(f"Loaded dataset with {self.length} samples. Mmap: {mmap_mode}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # With mmap, this reads from disk on demand
        input_emb = torch.from_numpy(np.array(self.txn_embeds[idx])).float()
        
        start = self.cat_offsets[idx]
        end = self.cat_offsets[idx+1]
        
        # Copy is necessary if using mmap to ensure we get a writable tensor if needed
        target_embs = torch.from_numpy(np.array(self.cat_embeds_flat[start:end])).float()
        
        return input_emb, target_embs

def collate_fn(batch):
    """
    Collate function to pad target embeddings and track actual lengths.
    """
    input_embs = []
    target_embs_list = []
    lengths = []
    
    # 1. Find max length
    max_len = 0
    for _, targets in batch:
        max_len = max(max_len, targets.size(0))
        
    # 2. Pad
    for input_emb, targets in batch:
        input_embs.append(input_emb)
        curr_len = targets.size(0)
        lengths.append(curr_len)
        
        if curr_len < max_len:
            # Pad with the last embedding
            last_emb = targets[-1].unsqueeze(0)
            padding = last_emb.repeat(max_len - curr_len, 1)
            padded_targets = torch.cat([targets, padding], dim=0)
            target_embs_list.append(padded_targets)
        else:
            target_embs_list.append(targets)
            
    input_stack = torch.stack(input_embs)
    target_stack = torch.stack(target_embs_list)
    lengths_stack = torch.tensor(lengths, dtype=torch.long)
    
    return input_stack, target_stack, lengths_stack