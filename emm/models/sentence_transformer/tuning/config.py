"""Configuration for sentence transformer tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class TuningConfig:
    """Configuration for fine-tuning sentence transformers
    
    Attributes:
        model_name: Base model to fine-tune
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        num_epochs: Number of training epochs
        precision: Training precision ('16-mixed', '32' etc)
        output_path: Where to save the model
        wandb_project: Optional W&B project for logging
        loss_type: Type of loss function ('dae', 'contrastive', 'combined')
        device_count: Number of devices to use
        similarity_threshold: Similarity threshold for filtering candidates
    """
    model_name: str = 'all-MiniLM-L6-v2'
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    precision: str = '16-mixed'
    output_path: Optional[Path] = None
    wandb_project: Optional[str] = None
    loss_type: str = 'dae'  # 'dae', 'contrastive', or 'combined'
    device_count: int = 1
    similarity_threshold: float = 0.5