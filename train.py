import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm
import logging
from torch import amp

from config import Config
from model import ModernTrajectoryNet
from dataset import TransactionCategoryDataset, collate_fn
from loss import *

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def calculate_metrics(output_emb, target_embs, lengths):
    """
    Returns:
        last_sim: Average similarity of the last item (Target: 1.0)
        monotonicity_acc: % of steps that strictly increase (Target: 100%)
    """
    with torch.no_grad():
        pred_norm = F.normalize(output_emb.float(), p=2, dim=1)
        targets_norm = F.normalize(target_embs.float(), p=2, dim=2)
        sims = torch.bmm(pred_norm.unsqueeze(1), targets_norm.transpose(1, 2)).squeeze(1) # [B, N]

        # 1. Check Last Item Similarity
        # gather the similarity at the actual last index
        batch_indices = torch.arange(sims.size(0), device=sims.device)
        last_indices = lengths - 1
        last_sim = sims[batch_indices, last_indices].mean().item()

        # 2. Check Monotonicity Accuracy
        # We want sim[i+1] > sim[i]
        # Create a mask for valid transitions (ignore padding)
        mask = torch.zeros_like(sims[:, :-1], dtype=torch.bool)
        for i, l in enumerate(lengths):
            if l > 1:
                mask[i, :l-1] = True
        
        diffs = sims[:, 1:] - sims[:, :-1]
        # Count how many valid steps are positive
        positive_steps = (diffs > 0) & mask
        
        # Avoid division by zero
        total_valid_steps = mask.sum()
        if total_valid_steps > 0:
            monotonicity_acc = positive_steps.sum().float() / total_valid_steps
            monotonicity_acc = monotonicity_acc.item()
        else:
            monotonicity_acc = 0.0
            
    return last_sim, monotonicity_acc


class ModelEMA:
    """Maintains an exponential moving average of model weights for evaluation stability."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def to(self, device):
        self.ema_model.to(device)
        return self

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer.detach())

def apply_augmentation(input_emb, p_mask=0.0, noise_std=0.0):
    """Applies masking and gaussian noise augmentations."""
    augmented = input_emb
    if p_mask > 0:
        mask = torch.rand_like(augmented) > p_mask
        augmented = augmented * mask.float()
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented

def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg.use_amp and device.type == "cuda"
    logger.info(f"Starting training on {device} with config: {cfg}")

    # --- Data ---
    logger.info("Loading dataset...")
    # Use mmap_mode='r' if your .npy files are larger than RAM
    dataset = TransactionCategoryDataset(data_dir=cfg.data_dir, mmap_mode='r')
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_data, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # --- Model & Utils ---
    model = ModernTrajectoryNet(cfg).to(device)
    ema_helper = None
    if 0 < cfg.ema_decay < 1.0:
        ema_helper = ModelEMA(model, cfg.ema_decay).to(device)
    
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            logger.info("Model compiled.")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")

    criterion = HybridTrajectoryLoss(monotonicity_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.scheduler_eta_min)
    scaler_device = device.type if device.type in ("cuda", "cpu") else "cuda"
    scaler = amp.GradScaler(scaler_device, enabled=amp_enabled)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # --- Training Loop ---
    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        
        for input_emb, target_embs, lengths in progress_bar:
            input_emb = input_emb.to(device)
            target_embs = target_embs.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            use_augmentation = cfg.augment_stop_epochs <= 0 or epoch < cfg.num_epochs - cfg.augment_stop_epochs
            model_input = apply_augmentation(input_emb, cfg.augment_p_mask, cfg.augment_noise_std) if use_augmentation else input_emb

            with amp.autocast(device_type=device.type, enabled=amp_enabled):
                output_emb = model(model_input)
                loss, acc, pos_sim, neg_sim = criterion(output_emb, target_embs, lengths)
            
            # Calculate Accuracy for monitoring
            with torch.no_grad():
                pred_norm = F.normalize(output_emb.float(), p=2, dim=1)
                targets_norm = F.normalize(target_embs.float(), p=2, dim=2)
                last_indices = lengths - 1
                true_last_targets = targets_norm[torch.arange(len(output_emb)), last_indices]
                logits = torch.matmul(pred_norm, true_last_targets.T)
                preds = torch.argmax(logits, dim=1)
                # acc = (preds == torch.arange(len(output_emb), device=preds.device)).float().mean()
            
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                optimizer.step()

            if ema_helper is not None:
                ema_helper.update(model)
            
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'L': f"{loss.item():.3f}", 
                'Pos': f"{pos_sim.item():.2f}",
                'Neg': f"{neg_sim.item():.2f}",
                'Acc': f"{acc.item():.1%}"
            })
                        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        eval_model = ema_helper.ema_model if ema_helper is not None else model
        eval_model.eval()
        val_loss = 0.0
        val_acc = 0.0 # Track validation accuracy
        sum_last_sim = 0.0
        sum_mono = 0.0
        metric_batches = 0
        
        with torch.no_grad():
            for input_emb, target_embs, lengths in val_loader:
                input_emb = input_emb.to(device)
                target_embs = target_embs.to(device)
                lengths = lengths.to(device)
                
                with amp.autocast(device_type=device.type, enabled=amp_enabled):
                    output_emb = eval_model(input_emb)
                    loss, batch_acc, _, _ = criterion(output_emb, target_embs, lengths)
                val_loss += loss.item()
                val_acc += batch_acc.item()

                last_sim, mono = calculate_metrics(output_emb, target_embs, lengths)
                sum_last_sim += last_sim
                sum_mono += mono
                metric_batches += 1
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader) # Average accuracy
        avg_last_sim = sum_last_sim / max(metric_batches, 1)
        avg_mono = sum_mono / max(metric_batches, 1)
        
        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {avg_val_acc:.2%} | LastSim: {avg_last_sim:.3f} | Mono: {avg_mono:.2%} | LR: {current_lr:.2e}"
        )

        # Save Best Model
        improved = (avg_val_loss + cfg.early_stop_min_delta) < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            save_checkpoint(model, optimizer, epoch, avg_val_loss, "checkpoints/best_model.pth")
            if ema_helper is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ema_helper.ema_model.state_dict(),
                    'loss': avg_val_loss,
                }, "checkpoints/best_model_ema.pth")
            logger.info("New best model saved.")
        else:
            epochs_no_improve += 1
            if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
                logger.info("Early stopping triggered based on validation loss.")
                break

if __name__ == "__main__":
    train()