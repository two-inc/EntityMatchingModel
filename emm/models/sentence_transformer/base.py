from __future__ import annotations

from typing import List, Dict, Optional, Union, Any
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Single numpy typing import with fallback
try:
    from numpy.typing import NDArray
except ImportError:
    from typing import Any
    NDArray = Any

from emm.models.sentence_transformer.utils import check_sentence_transformers_available

class BaseSentenceTransformerComponent:
    """Base component for sentence transformer functionality in EMM.
    
    This component provides shared functionality for both indexing and scoring
    using sentence transformers. It handles device management, batch size optimization,
    and common encoding operations.
    
    Attributes:
        device: The torch device being used ('cpu' or 'cuda')
        batch_size: Size of batches for encoding
        model: The underlying SentenceTransformer model
        model_kwargs: Additional kwargs passed to SentenceTransformer initialization
        encode_kwargs: Additional kwargs passed to encode() method
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the base sentence transformer component.
        
        Args:
            model_name: Name or path of the sentence transformer model to use
            device: Device to run model on ('cpu', 'cuda', or None for auto-detection)
            batch_size: Batch size for encoding (None for auto-detection)
            model_kwargs: Additional kwargs passed to SentenceTransformer initialization
            encode_kwargs: Additional kwargs passed to encode() method
            
        Example:
            >>> # Basic usage
            >>> base = BaseSentenceTransformerComponent()
            >>> 
            >>> # With custom settings including dimension truncation
            >>> base = BaseSentenceTransformerComponent(
            ...     model_name='mixedbread-ai/mxbai-embed-xsmall-v1',
            ...     device='cuda',
            ...     batch_size=64,
            ...     model_kwargs={'truncate_dim': 384}
            ... )
        """
        check_sentence_transformers_available()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        
        # Auto batch size detection
        if batch_size is None:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                self.batch_size = min(32, gpu_mem // (2**20))  # Conservative estimate
            else:
                self.batch_size = 32
        else:
            self.batch_size = batch_size
            
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.model = SentenceTransformer(model_name, device=self.device, **self.model_kwargs)
        self.model_name = model_name

    def encode_texts(self, texts: List[str]) -> NDArray[np.float32]:
        """Encode a list of texts into embeddings.
        
        This method handles batching and device management automatically.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
            
        Note:
            The returned embeddings are always on CPU and in numpy format
            for compatibility with scikit-learn and other tools.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            **self.encode_kwargs
        )
        return embeddings.cpu().numpy()

    def calculate_cosine_similarity(
        self, 
        embeddings1: NDArray[np.float32], 
        embeddings2: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Calculate cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings with shape (n_samples, embedding_dim)
            embeddings2: Second set of embeddings with shape (n_samples, embedding_dim)
            
        Returns:
            Array of cosine similarities with shape (n_samples,)
            
        Raises:
            ValueError: If embeddings have different shapes or unexpected dimensions
        """
        if embeddings1.shape != embeddings2.shape:
            raise ValueError(
                f"Embedding shapes must match: {embeddings1.shape} != {embeddings2.shape}"
            )
        
        # Convert to PyTorch tensors and move to correct device
        emb1 = torch.from_numpy(embeddings1).to(self.device)
        emb2 = torch.from_numpy(embeddings2).to(self.device)
        
        try:
            # Use sentence-transformers built-in cosine similarity
            similarities = util.cos_sim(emb1, emb2)
            # Extract diagonal for pairwise similarities
            result = similarities.diagonal().cpu().numpy()
            return result
        
        finally:
            # Clean up GPU memory
            if self.device.type == 'cuda':
                del emb1, emb2, similarities
                torch.cuda.empty_cache()

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory cache if using CUDA device."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache() 