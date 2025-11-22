from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 768
    n_layers: int = 8        # Deep enough to learn complex mappings
    expand: int = 4          # Standard expansion
    dropout: float = 0.2     # Increased for regularization
    drop_path_rate: float = 0.2 # Increased for regularization
    
    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2 # Stronger weight decay
    num_epochs: int = 40
    gradient_clip: float = 1.0
    use_amp: bool = True
    ema_decay: float = 0.999
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-3
    augment_p_mask: float = 0.1
    augment_noise_std: float = 0.02
    augment_stop_epochs: int = 5
    scheduler_eta_min: float = 1e-6
    
    # Data
    data_dir: str = "./dataset/gemma"
    num_workers: int = 4
    
    # Constraints
    sim_start: float = 0.0
    sim_end: float = 1.0