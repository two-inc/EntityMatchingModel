"""Tuning functionality for sentence transformers.

This module requires additional dependencies:
- sentence-transformers
- lightning
- wandb

Install with: pip install emm[tuning]
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import numpy as np

# Defer imports until actually needed
HAS_TUNING_DEPS = False
try:
    import torch
    import lightning as L
    import wandb
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler
    HAS_TUNING_DEPS = True
except ImportError:
    pass

from emm.models.sentence_transformer.tuning.config import TuningConfig

logger = logging.getLogger(__name__)

def check_tuning_dependencies():
    """Check if tuning dependencies are available"""
    if not HAS_TUNING_DEPS:
        raise ImportError(
            "sentence-transformers, torch, lightning, and wandb are required for tuning. "
            "Install with: pip install emm[tuning]"
        )

class SentenceTransformerTuner:
    """Fine-tuning for sentence transformers specialized for company name matching"""
    
    def __init__(self, config: TuningConfig):
        """Initialize tuner with configuration
        
        Args:
            config: Tuning configuration object
        """
        check_tuning_dependencies()
        self.config = config
        
        # Setup Lightning Fabric for distributed training
        self.fabric = L.Fabric(
            accelerator="auto",
            devices=config.device_count,
            precision=config.precision,
            strategy="ddp" if config.device_count > 1 else "auto"
        )
        
        # Initialize model with efficient settings
        self.model = SentenceTransformer(
            config.model_name,
            device=self.fabric.device,
            cache_folder=config.output_path / ".cache" if config.output_path else None
        )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.precision == "16-mixed" else None
        
        # Initialize tracking metrics
        self.best_loss = float('inf')
        self.best_model_path = None
        
    def _setup_loss(self) -> torch.nn.Module:
        """Initialize loss function with optimizations"""
        if self.config.loss_type == 'dae':
            return losses.DenoisingAutoEncoderLoss(
                self.model,
                decoder_name_or_path=self.config.model_name,  # Reuse encoder weights
                tie_encoder_decoder=True  # Memory efficient
            )
        elif self.config.loss_type == 'contrastive':
            return losses.ContrastiveTensionLoss(
                self.model,
                distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
                margin=0.5
            )
        elif self.config.loss_type == 'combined':
            return self._setup_combined_loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
            
    def _setup_combined_loss(self) -> torch.nn.Module:
        """Setup memory-efficient combined loss"""
        class CombinedLoss(torch.nn.Module):
            def __init__(self, model, dae_weight=1.0, contrastive_weight=0.5):
                super().__init__()
                self.dae = losses.DenoisingAutoEncoderLoss(
                    model,
                    decoder_name_or_path=model.get_config_dict()['model_name'],
                    tie_encoder_decoder=True
                )
                self.contrastive = losses.ContrastiveTensionLoss(
                    model,
                    distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE
                )
                self.dae_weight = dae_weight
                self.contrastive_weight = contrastive_weight
                
            def forward(self, batch):
                dae_loss = self.dae(batch)
                contrastive_loss = self.contrastive(batch)
                return self.dae_weight * dae_loss + self.contrastive_weight * contrastive_loss
                
        return CombinedLoss(self.model)
        
    def setup_training(self, train_dataloader: DataLoader) -> None:
        """Setup model, optimizer and dataloader with optimizations"""
        self.loss_fn = self._setup_loss()
        
        # Use AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),  # Standard betas
            eps=1e-8
        )
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_dataloader) * self.config.num_epochs
        )
        
        # Setup with Fabric
        self.model, self.optimizer = self.fabric.setup(
            self.model, 
            self.optimizer
        )
        self.train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint efficiently"""
        if self.config.output_path and loss < self.best_loss:
            self.best_loss = loss
            checkpoint_path = self.config.output_path / f"checkpoint-epoch-{epoch}"
            
            # Save in efficient safetensors format if available
            try:
                self.model.save(
                    checkpoint_path,
                    save_optimizer_state=True,
                    safe_serialization=True
                )
            except Exception:
                self.model.save(checkpoint_path)
                
            self.best_model_path = checkpoint_path
            
    def train(self) -> None:
        """Execute optimized training loop"""
        if self.config.wandb_project and self.fabric.is_global_zero:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )
            
        self.fabric.launch()
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0.0
            
            # Use tqdm for progress tracking
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Mixed precision training
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = self.loss_fn(batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.loss_fn(batch)
                    self.fabric.backward(loss)
                    self.optimizer.step()
                
                # Update learning rate
                self.scheduler.step()
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Log metrics
                if self.fabric.is_global_zero and self.config.wandb_project:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            
            # Calculate average loss
            avg_loss = total_loss / len(self.train_dataloader)
            
            # Save checkpoint
            self.save_checkpoint(epoch, avg_loss)
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            if self.fabric.is_global_zero and self.config.wandb_project:
                wandb.log({
                    "epoch_loss": avg_loss,
                    "epoch": epoch
                })
        
        # Save final model
        if self.config.output_path:
            final_path = self.config.output_path / "final-model"
            self.model.save(final_path, safe_serialization=True)
            
    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Efficient encoding with the tuned model"""
        batch_size = batch_size or self.config.batch_size
        
        try:
            with torch.inference_mode():
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model.encode(
                            texts,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            convert_to_tensor=True,
                            normalize_embeddings=True
                        )
                else:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                return embeddings.cpu().numpy()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()