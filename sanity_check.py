import torch
import torch.optim as optim
from model import HybridMambaAttentionModel
from dataset import TransactionCategoryDataset, collate_fn
from torch.utils.data import DataLoader
from loss import InfoNCEMonotonicLoss

def sanity_check():
    # 1. Config: ZERO Regularization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with NO dropout
    # Ensure your config/model has dropout=0.0
    from config import Config
    cfg = Config()
    cfg.dropout = 0.0 
    
    model = HybridMambaAttentionModel(cfg).to(device)
    model.train()
    
    # 2. Get ONE batch
    dataset = TransactionCategoryDataset(data_dir=cfg.data_dir, mmap_mode='r')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    input_emb, target_embs, lengths = next(iter(loader))
    
    input_emb = input_emb.to(device)
    target_embs = target_embs.to(device)
    lengths = lengths.to(device)
    
    # 3. Aggressive Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0) # High LR, No Decay
    criterion = InfoNCEMonotonicLoss(temperature=0.1) # Standard temp

    print("--- Starting Overfit Sanity Check ---")
    print("Goal: Loss < 0.01")
        
    # We need to see if Accuracy is 100% even if Loss is 1.45
    for i in range(200): # Increased to 200 iterations
        optimizer.zero_grad()
        output_emb = model(input_emb)
        
        # --- Manual Accuracy Calculation ---
        # Replicating the InfoNCE logic to check matches
        pred_norm = torch.nn.functional.normalize(output_emb, p=2, dim=1)
        targets_norm = torch.nn.functional.normalize(target_embs, p=2, dim=2)
        
        # Get the last target for every item
        batch_indices = torch.arange(output_emb.size(0), device=device)
        last_indices = lengths - 1
        true_last_targets = targets_norm[batch_indices, last_indices, :]
        
        # Dot product [B, D] @ [D, B] -> [B, B]
        logits = torch.matmul(pred_norm, true_last_targets.transpose(0, 1))
        preds = torch.argmax(logits, dim=1)
        
        # Accuracy: How many times did diagonal win?
        acc = (preds == batch_indices).float().mean().item()
        # ------------------------------------

        loss = criterion(output_emb, target_embs, lengths)
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: Loss {loss.item():.4f} | Accuracy: {acc*100:.1f}%")
        
        if i % 10 == 0:
            print(f"Iter {i}: Loss {loss.item():.5f}")

if __name__ == "__main__":
    sanity_check()