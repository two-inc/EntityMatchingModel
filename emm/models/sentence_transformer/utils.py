def check_sentence_transformers_available():
    """Check if sentence-transformers is available and raise a consistent error if not."""
    try:
        import sentence_transformers
        return True
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Please install it with `pip install emm[transformers]` "
            "or `pip install sentence-transformers`"
        ) 