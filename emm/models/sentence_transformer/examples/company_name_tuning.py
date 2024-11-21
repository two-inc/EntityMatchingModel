from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from emm.models.sentence_transformer.tuning import SentenceTransformerTuner, TuningConfig

class CompanyNameDataset(Dataset):
    """Dataset for company name tuning that creates pairs for training"""
    
    def __init__(self, company_names: List[str]):
        self.company_names = company_names
        
    def __len__(self):
        return len(self.company_names)
        
    def __getitem__(self, idx):
        # For DAE loss, we just need the original text
        # The loss function handles text corruption internally
        return self.company_names[idx]

def create_company_name_dataloader(
    company_names: List[str],
    batch_size: int = 32
) -> DataLoader:
    """Create dataloader for company name tuning"""
    dataset = CompanyNameDataset(company_names)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

def tune_company_names(
    company_names: List[str],
    output_dir: Path,
    **config_kwargs
) -> SentenceTransformerTuner:
    """Fine-tune a sentence transformer model for company name matching
    
    Args:
        company_names: List of company names to train on
        output_dir: Directory to save the model
        **config_kwargs: Optional overrides for TuningConfig
        
    Returns:
        Trained SentenceTransformerTuner instance
    """
    # Configure tuning with defaults and any overrides
    config = TuningConfig(
        model_name='all-MiniLM-L6-v2',
        batch_size=32,
        num_epochs=3,
        loss_type='dae',  # DAE works well for company names
        output_path=output_dir / 'company_name_model',
        wandb_project='company-name-matching',
        **config_kwargs
    )
    
    # Create dataloader
    train_dataloader = create_company_name_dataloader(
        company_names,
        batch_size=config.batch_size
    )
    
    # Initialize and train
    tuner = SentenceTransformerTuner(config)
    tuner.setup_training(train_dataloader)
    tuner.train()
    
    return tuner

def example_usage():
    """Example of how to use the company name tuner"""
    # Sample company names
    company_names = [
        "Apple Inc.",
        "Apple Corporation",
        "Microsoft Corporation",
        "Microsoft Corp.",
        "Google LLC",
        "Alphabet Inc.",
        # ... add more examples
    ]
    
    output_dir = Path("models/company_names")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train with custom config
    tuner = tune_company_names(
        company_names,
        output_dir,
        num_epochs=5,  # Override default epochs
        learning_rate=1e-5  # Override default learning rate
    )
    
    # Example: encode some company names
    embeddings = tuner.encode([
        "Apple Incorporated",
        "Microsoft Corporation"
    ])
    
    return embeddings

if __name__ == "__main__":
    example_usage()