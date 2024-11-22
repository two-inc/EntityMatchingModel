"""Example of entity matching using sentence transformers for indexing.

This example requires sentence-transformers.
Install with: pip install emm[transformers]
"""

# Check required dependencies at import time
try:
    import pandas as pd
    from emm import PandasEntityMatching
except ImportError as e:
    raise ImportError(
        "This example requires sentence-transformers. "
        "Install with: pip install emm[transformers]"
    ) from e

def main():
    # Create sample ground truth data
    gt = pd.DataFrame([
        (1, 'John Smith LLC'),
        (2, 'ING LLC'),
        (3, 'John Doe LLC'),
        (4, 'Zhe Sun G.M.B.H'),
        (5, 'Random GMBH'),
    ], columns=['id', 'name'])

    print("Ground truth data:")
    print(gt)
    print()

    # Initialize entity matcher with sentence transformer indexer
    # Using mixedbread-ai's optimized model with truncated dimensions
    nm = PandasEntityMatching({
        'name_only': True,
        'entity_id_col': 'id',
        'name_col': 'name',
        'preprocessor': 'preprocess_merge_abbr',
        'indexers': [{
            'type': 'sentence_transformer',
            'model_name': 'mixedbread-ai/mxbai-embed-xsmall-v1',
            'num_candidates': 5,
            'similarity_threshold': 0.5,
            # Pass model-specific parameters through model_kwargs
            'model_kwargs': {
                'truncate_dim': 384,  # Recommended dimension from model docs
                'normalize_embeddings': True  # Optional: ensure normalized embeddings
            },
            # Optional: customize encoding behavior
            'encode_kwargs': {
                'normalize_embeddings': True
            }
        }],
        'supervised_on': False,
    })

    # Fit the indexer to ground truth data
    print("Fitting model to ground truth...")
    nm.fit(gt)

    # Create test data with some variations
    test_data = pd.DataFrame([
        (10, 'John Smith'),         # Should match John Smith LLC
        (11, 'I.n.G. LLC'),        # Should match ING LLC
        (12, 'Jon DOEE LLC'),      # Should approximately match John Doe LLC
        (13, 'Random Company'),     # May not find good matches
    ], columns=['id', 'name'])

    print("\nTest data:")
    print(test_data)
    print()

    # Perform matching
    print("Performing name matching...")
    results = nm.transform(test_data)

    print("\nMatching results:")
    print(results)

    # Print some statistics
    print("\nMatching statistics:")
    print(f"Total test cases: {len(test_data)}")
    print(f"Total matches found: {len(results)}")
    print(f"Average similarity score: {results['nm_score'].mean():.3f}")

if __name__ == "__main__":
    main() 