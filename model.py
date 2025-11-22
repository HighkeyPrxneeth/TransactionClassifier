import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable and computationally efficient than LayerNorm.
    Used in LLaMA, PaLM, Gopher.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    SOTA activation function for FFNs (outperforms GELU/ReLU).
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gate mechanism: (x * sigmoid(x)) * linear(x)
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(self.dropout(hidden))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Allows the model to dynamically weight different dimensions of the embedding
    based on global context.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input: [B, D] -> unsqueeze to [B, D, 1] for pool/conv compatibility if needed
        # But here we are working with vectors, so we simulate it.
        b, d = x.shape
        y = self.fc(x) # [B, D]
        return x * y

class DropPath(nn.Module):
    """Stochastic depth regularizer (Improved)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ModernBlock(nn.Module):
    """
    A Pre-Norm Block combining RMSNorm, SwiGLU, and Channel Attention.
    """
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1, 
                 layer_scale_init: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        
        # 1. Normalization
        self.norm = RMSNorm(dim)
        
        # 2. SOTA Feed Forward (SwiGLU)
        # SwiGLU usually requires 2/3 hidden dim of standard MLP to match params, 
        # but we keep it high for expressivity.
        self.ffn = SwiGLU(dim, int(dim * expand * 2 / 3), dropout=dropout)
        
        # 3. Channel Attention (Context awareness)
        self.se = SEBlock(dim, reduction=4)
        
        # 4. Regularization
        self.layer_scale = nn.Parameter(torch.ones(dim) * layer_scale_init) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        
        # Pre-Norm Architecture
        out = self.norm(x)
        out = self.ffn(out)
        out = self.se(out) # Apply attention
        
        if self.layer_scale is not None:
            out = out * self.layer_scale
            
        out = self.drop_path(out)
        
        return residual + out

class ModernTrajectoryNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        
        # Config defaults
        dropout = getattr(config, "dropout", 0.1)
        expand = getattr(config, "expand", 4)
        drop_path_rate = getattr(config, "drop_path_rate", 0.1)
        
        # Input Projection (Projects to latent space)
        self.input_proj = nn.Sequential(
            RMSNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Backbone
        self.blocks = nn.ModuleList([
            ModernBlock(
                dim=self.d_model,
                expand=expand,
                dropout=dropout,
                drop_path=drop_path_rate * (i / (self.n_layers - 1)) # Linear decay
            ) for i in range(self.n_layers)
        ])
        
        self.final_norm = RMSNorm(self.d_model)
        
        # Projector Head (SimCLR / CLIP style)
        # Important: Keep high dimension for the final linear probe
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_trajectory=False):
        # Handle sequence dimension if present
        if x.dim() == 3:
            x = x.mean(dim=1)
            
        x = self.input_proj(x)
        
        trajectory = []
        for block in self.blocks:
            x = block(x)
            trajectory.append(x)
            
        x = self.final_norm(x)
        
        # Residual connection to original input is implicit via the blocks,
        # but for trajectory learning, we want the final head to dictate the shift.
        output = self.head(x)
        
        # OPTIONAL: Add Denoising / Residual connection to input
        # output = output + input_tensor_if_saved
        
        if return_trajectory:
            return output, torch.stack(trajectory, dim=1)
            
        return output

# Backwards compatibility
HybridMambaAttentionModel = ModernTrajectoryNet
