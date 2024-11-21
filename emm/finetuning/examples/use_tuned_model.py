from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_tuned_model(model_path: Path | str):
    """Load a fine-tuned model from local storage"""
    return SentenceTransformer(model_path)

# Example usage
if __name__ == "__main__":
    # Load the model
    model_path = Path("./company_name_model")  # Path to saved model
    model = load_tuned_model(model_path)
    
    # Use for encoding
    companies = [
        "Microsoft Corporation Ltd",
        "MSFT Corp Limited",
        "Apple Inc",
        "AAPL Corporation"
    ]
    
    # Get embeddings
    embeddings = model.encode(
        companies,
        batch_size=32,
        show_progress_bar=False,
        convert_to_tensor=True
    )
    
    # Calculate similarities
    from torch.nn.functional import cosine_similarity
    similarity = cosine_similarity(
        embeddings[0].unsqueeze(0),  # Microsoft
        embeddings[1].unsqueeze(0)    # MSFT
    )
    print(f"Similarity score: {similarity.item():.4f}") 