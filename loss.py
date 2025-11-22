import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IncreasingSimilarityLoss(nn.Module):
    """
    MSE Loss that forces cosine similarity to follow a specific trajectory 
    (e.g., 0.95 -> 1.0) over the sequence length.
    """
    def __init__(self, start_val=0.95, end_val=1.0):
        super().__init__()
        self.start = start_val
        self.end = end_val

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        """
        Args:
            predicted_embedding: [B, d_model]
            target_embeddings_list: [B, max_len, d_model]
            actual_lengths: [B] (tensor containing actual sequence lengths)
        """
        batch_size, max_len, _ = target_embeddings_list.shape
        device = predicted_embedding.device

        # Normalize inputs
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)       # [B, 1, D]
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2) # [B, N, D]

        # Compute similarities: [B, N]
        # bmm: (B, 1, D) x (B, D, N) -> (B, 1, N) -> squeeze -> (B, N)
        similarities = torch.bmm(pred_norm.unsqueeze(1), targets_norm.transpose(1, 2)).squeeze(1)

        # Generate dynamic ideal curves for each item in batch
        # We construct a mask to calculate loss only on valid items (and valid pads)
        ideal_matrix = torch.zeros((batch_size, max_len), device=device)
        
        for i in range(batch_size):
            length = int(actual_lengths[i].item())
            if length > 0:
                # Generate curve 0.95 -> 1.0 over 'length' steps
                curve = torch.linspace(self.start, self.end, steps=length, device=device)
                ideal_matrix[i, :length] = curve
                # If padded, the target remains 1.0 (perfect match) for the padding
                if length < max_len:
                    ideal_matrix[i, length:] = self.end
        
        # MSE Loss
        loss = F.mse_loss(similarities, ideal_matrix)
        return loss
    
# loss.py
class MonotonicRankingLoss(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2)
        
        # [B, N]
        sims = torch.bmm(pred_norm.unsqueeze(1), targets_norm.transpose(1, 2)).squeeze(1)
        
        # Calculate loss: Sim[i+1] should be > Sim[i] + margin
        # We compare sim[:, :-1] (current) with sim[:, 1:] (next)
        
        current_step = sims[:, :-1]
        next_step = sims[:, 1:]
        
        # Relu(Current - Next + Margin)
        # If Next > Current + Margin, result is 0 (Good).
        # If Next is small, result is positive (Loss).
        losses = F.relu(current_step - next_step + self.margin)
        
        # Add a term to maximize the LAST similarity (to ensure we don't just output 0.0 everywhere)
        # We want the LAST valid item to be close to 1.0
        last_indices = actual_lengths - 1
        last_sims = sims[torch.arange(sims.size(0)), last_indices]
        loss_last = 1.0 - last_sims.mean()
        
        return losses.mean() + loss_last

class ContrastiveMonotonicLoss(nn.Module):
    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        # --- Setup ---
        batch_size = predicted_embedding.size(0)
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2)
        
        # [B, N]
        sims = torch.bmm(pred_norm.unsqueeze(1), targets_norm.transpose(1, 2)).squeeze(1)
        
        # --- Component 1: Monotonicity & Last Target ---
        # Ensure increasing similarity
        diffs = sims[:, :-1] - sims[:, 1:] + 0.01 # Small margin for monotonicity
        loss_monotonic = F.relu(diffs).mean()
        
        # Ensure last target is close to 1.0
        batch_indices = torch.arange(batch_size, device=predicted_embedding.device)
        last_indices = actual_lengths - 1
        last_sims = sims[batch_indices, last_indices]
        loss_pos = (1.0 - last_sims).mean()

        # --- Component 2: Negative Mining (The Fix) ---
        # We want: Sim(Input, True_Target) > Sim(Input, Random_Wrong_Target) + Margin
        
        # Trick: We can use other elements in the batch as negatives!
        # Calculate sim between Input[i] and Last_Target[j] where i != j
        
        # Get the "Last Target" for every item in the batch -> [B, D]
        last_targets = targets_norm[batch_indices, last_indices, :] 
        
        # Matrix multiplication of all Inputs vs all Last Targets -> [B, B]
        # all_sims[i, j] = similarity of Input i with Target j
        all_sims = torch.mm(pred_norm, last_targets.t())
        
        # Mask out the diagonal (positive pairs)
        mask = torch.eye(batch_size, device=predicted_embedding.device).bool()
        
        # Get the positive similarity (diagonal)
        pos_sim = all_sims.diag()
        
        # Get the hardest negative for each row (max sim that isn't the diagonal)
        # Fill diagonal with -1 so it doesn't get picked as max
        all_sims.masked_fill_(mask, -1.0)
        neg_sim, _ = all_sims.max(dim=1)
        
        # Triplet Loss: Pos > Neg + Margin
        loss_contrastive = F.relu(neg_sim - pos_sim + self.margin).mean()
        
        return loss_monotonic + loss_pos + loss_contrastive
    
class InfoNCEMonotonicLoss(nn.Module):
    def __init__(self, temperature=0.05, monotonicity_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.mono_weight = monotonicity_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        """
        Args:
            predicted_embedding: [B, D]
            target_embeddings_list: [B, Max_Len, D]
            actual_lengths: [B]
        """
        batch_size = predicted_embedding.size(0)
        device = predicted_embedding.device
        
        # 1. Normalize everything
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2)
        
        # Extract the correct last embedding for each item in batch
        # [B, D]
        last_indices = actual_lengths - 1
        batch_indices = torch.arange(batch_size, device=device)
        true_last_targets = targets_norm[batch_indices, last_indices, :]
        
        # Compute Similarity Matrix: [Batch, Batch]
        # Row i, Col j = Sim(Prediction i, Target j)
        logits = torch.matmul(pred_norm, true_last_targets.transpose(0, 1))
        
        # Scale by temperature (makes distribution sharper)
        logits = logits / self.temperature
        
        # The label for row i is index i (the diagonal is the correct match)
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        loss_contrastive = self.ce_loss(logits, labels)
        
        # Calculate similarities against OWN sequence: [B, Max_Len]
        # unsqueeze(1) -> [B, 1, D]
        # transpose(1,2) -> [B, D, Max_Len]
        # bmm -> [B, 1, Max_Len] -> squeeze -> [B, Max_Len]
        own_sims = torch.bmm(pred_norm.unsqueeze(1), targets_norm.transpose(1, 2)).squeeze(1)
        
        # Calculate difference: Sim[i+1] - Sim[i]
        # We want this to be POSITIVE.
        # We penalize if Sim[i] > Sim[i+1]
        current_step = own_sims[:, :-1]
        next_step = own_sims[:, 1:]
        
        # Mask out padding (we don't care about increasing similarity in padding)
        mask = torch.zeros_like(current_step, dtype=torch.bool)
        for i, l in enumerate(actual_lengths):
            if l > 1:
                mask[i, :l-1] = True
        
        # Loss = ReLU(Current - Next + margin)
        # We enforce a small positive slope
        diffs = current_step - next_step + 0.005
        loss_mono = (F.relu(diffs) * mask).sum() / (mask.sum() + 1e-8)
        
        return loss_contrastive + (loss_mono * self.mono_weight)
    
class SupervisedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        """
        SupCon Loss adapted for your sequence setup.
        """
        device = predicted_embedding.device
        batch_size = predicted_embedding.shape[0]
        
        # Normalize
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2)
        
        # Get the True Last Target for every item
        # [B, D]
        last_indices = actual_lengths - 1
        batch_indices = torch.arange(batch_size, device=device)
        true_last_targets = targets_norm[batch_indices, last_indices, :]
        
        # Sim(Pred, All_Targets) -> [B, B]
        logits = torch.matmul(pred_norm, true_last_targets.T) / self.temperature
        
        # We need to know which items in the batch have the EXACT SAME target embedding.
        # Compute Sim(Target, Target) -> [B, B]
        target_sim_matrix = torch.matmul(true_last_targets, true_last_targets.T)
        
        # If sim > 0.99, they are the same target class.
        # Mask: 1 if same class, 0 if different.
        labels_mask = (target_sim_matrix > 0.99).float()
        
        # Remove self-contrast (diagonal) from the mask for the denominator
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() # Stability trick
        
        # Compute exp
        exp_logits = torch.exp(logits)
        
        # Denominator: Sum of all exp_logits *except* the positive matches is standard,
        # but in SupCon, denominator is usually sum over all excluding self.
        # Simpler approach: LogSoftmax formulation
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Mean of log_likelihood over positive matches
        # We want to maximize log_prob for all j where labels_mask[i,j] == 1
        mean_log_prob_pos = (labels_mask * log_prob).sum(1) / (labels_mask.sum(1) + 1e-9)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
class ContrastiveMarginLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        """
        Returns: (loss, accuracy, avg_pos_sim, avg_neg_sim)
        """
        device = predicted_embedding.device
        batch_size = predicted_embedding.size(0)

        # 1. Normalize
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2)

        # 2. Get the True Last Target for each item
        # [B, D]
        last_indices = actual_lengths - 1
        batch_indices = torch.arange(batch_size, device=device)
        true_last_targets = targets_norm[batch_indices, last_indices, :]

        # 3. Calculate Logits [B, B]
        # row i, col j = Similarity between Pred[i] and Target[j]
        logits = torch.matmul(pred_norm, true_last_targets.T)
        
        # Metrics for logging
        # Diagonal terms are the Positive Similarities
        pos_sim = logits.diag().mean()
        
        # Off-diagonal terms are the Negative Similarities
        mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
        neg_sim = logits[mask].mean()
        
        # Calculate Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch_indices).float().mean()

        # 4. Compute Loss
        # The target for row i is index i (the diagonal)
        scaled_logits = logits / self.temperature
        loss = self.criterion(scaled_logits, batch_indices)

        return loss, acc, pos_sim, neg_sim

class AdaptiveInfoNCELoss(nn.Module):
    """
    InfoNCE Loss with a learnable temperature parameter.
    """
    def __init__(self, init_temperature=0.07, min_temp=0.01, max_temp=1.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        device = predicted_embedding.device
        batch_size = predicted_embedding.size(0)
        
        # Clamp temperature to avoid instability
        self.logit_scale.data.clamp_(min=np.log(1/self.max_temp), max=np.log(1/self.min_temp))
        temperature = torch.exp(-self.logit_scale)
        
        # Normalize
        pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
        targets_norm = F.normalize(target_embeddings_list, p=2, dim=2)
        
        # Get True Last Targets
        last_indices = actual_lengths - 1
        batch_indices = torch.arange(batch_size, device=device)
        true_last_targets = targets_norm[batch_indices, last_indices, :]
        
        # Logits [B, B]
        logits = torch.matmul(pred_norm, true_last_targets.T) * torch.exp(self.logit_scale)
        
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        loss = self.ce_loss(logits, labels)
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
            # Raw similarities for logging
            raw_sims = torch.matmul(pred_norm, true_last_targets.T)
            pos_sim = raw_sims.diag().mean()
            mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
            neg_sim = raw_sims[mask].mean()
        
        return loss, acc, pos_sim, neg_sim

class HybridTrajectoryLoss(nn.Module):
    """
    Combines Adaptive InfoNCE (for final target alignment) 
    with Monotonic Ranking (for trajectory consistency).
    """
    def __init__(self, monotonicity_weight=1.0):
        super().__init__()
        self.contrastive = AdaptiveInfoNCELoss()
        self.monotonic = MonotonicRankingLoss(margin=0.01)
        self.mono_weight = monotonicity_weight

    def forward(self, predicted_embedding, target_embeddings_list, actual_lengths):
        loss_con, acc, pos_sim, neg_sim = self.contrastive(predicted_embedding, target_embeddings_list, actual_lengths)
        loss_mono = self.monotonic(predicted_embedding, target_embeddings_list, actual_lengths)
        
        return loss_con + (loss_mono * self.mono_weight), acc, pos_sim, neg_sim