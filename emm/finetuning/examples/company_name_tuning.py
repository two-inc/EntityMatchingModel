from emm.finetuning.sentence_transformer_tuner import SentenceTransformerTuner, TuningConfig
from pathlib import Path

def tune_company_names(company_names, output_dir: Path):
    # Configure tuning
    config = TuningConfig(
        model_name='all-MiniLM-L6-v2',
        batch_size=32,
        num_epochs=3,
        loss_type='dae',
        output_path=output_dir / 'company_name_model',
        wandb_project='company-name-matching'
    )
    
    # Create dataloader
    train_dataloader = create_company_name_dataloader(company_names)
    
    # Initialize and train
    tuner = SentenceTransformerTuner(config)
    tuner.setup_training(train_dataloader)
    tuner.train()
    
    return tuner 