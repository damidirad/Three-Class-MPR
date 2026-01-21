from dataclasses import dataclass
from typing import List
import pathlib

@dataclass
class Config:
    """
    Hyperparameter configuration for MPR with SST-based fairness regularization.
    """
    # Model architecture
    emb_size: int = 64
    sst_hidden_sizes: List[int] = None
    dropout_rate: float = 0.1

    # General settings
    seed: int = 1
    s_attr: str = "gender"
    task_type: str = "Lastfm-360K"
    gpu_id: int = 7 # set to 0 if only one GPU is available
    unfair_model: str = "./pretrained_model/Lastfm-360K/MF_orig_model" # use pathlib
    early_stopping_patience: int = 10
    
    # Training hyperparameters
    sst_epochs: int = 50
    mf_epochs: int = 200
    sst_lr: float = 1e-4
    mf_lr: float = 1e-3
    batch_size: int = 32768
    weight_decay: float = 1e-7
    
    # Fairness parameters
    beta: float = 0.005
    fair_reg: float = 12.0
    alpha_max: float = 0.3
    warmup_epochs: int = 20
    
    # Prior selection
    num_active_priors: int = 14
    full_sweep_interval: int = 10
    prior_focus_window: int = 5  # focus on worst priors
    
    # Alternating optimization
    alternating_interval: int = 15  # update SST every n epochs
    sst_refinement_steps: int = 5
    
    # Numerical stability
    eps: float = 1e-6
    grad_clip: float = 1.0
    weight_clip_min: float = 0.1
    weight_clip_max: float = 10.0
    
    # Stalling detection
    stall_window: int = 8
    stall_tolerance: float = 1e-3

    # Sensitive attribute ratios
    s0_ratio: float = 0.5
    s1_ratio: float = 0.1 # -> 0.5
    s2_ratio: float = 0.1 

    # Evaluation
    evaluation_interval: int = 3
    
    def __post_init__(self):
        if self.sst_hidden_sizes is None:
            self.sst_hidden_sizes = [128, 64]

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}