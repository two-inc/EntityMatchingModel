from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
import lightning as L
from sentence_transformers import SentenceTransformer, losses
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

from emm.models.sentence_transformer.tuning.config import TuningConfig

logger = logging.getLogger(__name__)

class SentenceTransformerTuner:
    """Fine-tuning for sentence transformers specialized for company name matching"""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.fabric = L.Fabric(
            accelerator="auto",
            devices=config.device_count,
            precision=config.precision
        )
        
        self.model = SentenceTransformer(config.model_name)
        
    def _setup_loss(self) -> torch.nn.Module:
        """Initialize loss function based on config"""
        if self.config.loss_type == 'dae':
            return losses.DenoisingAutoEncoderLoss(self.model)
        elif self.config.loss_type == 'contrastive':
            return losses.ContrastiveTensionLoss(self.model)
        elif self.config.loss_type == 'combined':
            return self._setup_combined_loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
            
    def _setup_combined_loss(self) -> torch.nn.Module:
        """Setup combined DAE and contrastive loss"""
        class CombinedLoss(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.dae = losses.DenoisingAutoEncoderLoss(model)
                self.contrastive = losses.ContrastiveTensionLoss(model)
                
            def forward(self, batch):
                return self.dae(batch) + 0.5 * self.contrastive(batch)
                
        return CombinedLoss(self.model)
        
    def setup_training(self, train_dataloader: DataLoader) -> None:
        """Setup model, optimizer and dataloader"""
        self.loss_fn = self._setup_loss()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup with Fabric
        self.model, self.optimizer = self.fabric.setup(
            self.model, 
            self.optimizer
        )
        self.train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        
    def train(self) -> None:
        """Execute training loop"""
        if self.config.wandb_project and self.fabric.is_global_zero:
            try:
                wandb.init(project=self.config.wandb_project)
            except Exception as e:
                logger.warning(f"Failed to initialize W&B logging: {e}")
                
        self.fabric.launch()
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(batch)
                self.fabric.backward(loss)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if self.fabric.is_global_zero and self.config.wandb_project:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch
                    })
            
            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            
        if self.config.output_path:
            self.model.save(self.config.output_path)
            
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Utility method to encode texts with fine-tuned model"""
        return self.model.encode(texts)