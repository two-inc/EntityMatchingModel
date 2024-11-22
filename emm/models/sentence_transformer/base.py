"""Base sentence transformer component for entity matching.

This module requires sentence-transformers and torch.
Install with: pip install emm[transformers]
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, Tuple
import logging

# Check required dependencies at import time
try:
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer, util, SimilarityFunction
except ImportError as e:
    raise ImportError(
        "sentence-transformers and torch are required for this module. "
        "Install with: pip install emm[transformers]"
    ) from e

# Rest of imports
from functools import lru_cache
import numpy as np

# Single numpy typing import with fallback
try:
    from numpy.typing import NDArray
except ImportError:
    from typing import Any
    NDArray = Any

logger = logging.getLogger(__name__)

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
        similarity_threshold: float = 0.5,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        cache_size: int = 10000,
        use_mixed_precision: bool = True,
    ) -> None:
        """Initialize the base sentence transformer component.
        
        Args:
            model_name: Name or path of the sentence transformer model to use
            similarity_threshold: Similarity threshold for filtering candidates
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
        self.similarity_threshold = similarity_threshold
        
        # Setup caching for frequently accessed texts
        self._setup_cache(cache_size)
        
        # Use dot product for normalized embeddings
        self.model_kwargs.setdefault('similarity_fn_name', SimilarityFunction.DOT_PRODUCT)
        # Enable normalization during encoding
        self.encode_kwargs.setdefault('normalize_embeddings', True)
        
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def _setup_cache(self, cache_size: int) -> None:
        """Setup caching for embeddings"""
        @lru_cache(maxsize=cache_size)
        def cached_encode(text: str) -> Tuple[float, ...]:
            # Convert single text to embedding and cache as tuple
            embedding = self.model.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_tensor=True,
                **self.encode_kwargs
            )
            return tuple(embedding.cpu().numpy().flatten())
            
        self._cached_encode = cached_encode

    def encode_texts(self, texts: List[str]) -> NDArray[np.float32]:
        """Encode with optional mixed precision"""
        if len(texts) == 1:
            return np.array([self._cached_encode(texts[0])])

        first_encoding = self.model.encode(
            [texts[0]], 
            batch_size=1,
            show_progress_bar=False,
            convert_to_tensor=True,
            **self.encode_kwargs
        )
        
        embedding_dim = first_encoding.shape[1]
        output = np.zeros((len(texts), embedding_dim), dtype=np.float32)
        output[0] = first_encoding.cpu().numpy()

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    with torch.inference_mode():
                        embeddings = self.model.encode(
                            batch_texts,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_tensor=True,
                            **self.encode_kwargs
                        )
            else:
                with torch.inference_mode():
                    embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        **self.encode_kwargs
                    )
                    
            output[i:i + len(batch_texts)] = embeddings.cpu().numpy()
            self._cleanup_gpu_tensors([embeddings])

        return output

    @property
    def cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        info = self._cached_encode.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._cached_encode.cache_clear()

    def calculate_cosine_similarity(
        self, 
        embeddings1: NDArray[np.float32], 
        embeddings2: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Calculate cosine similarity between two sets of embeddings efficiently."""
        if embeddings1.shape != embeddings2.shape:
            raise ValueError(
                f"Embedding shapes must match: {embeddings1.shape} != {embeddings2.shape}"
            )
        
        # Convert and normalize on CPU first if possible
        emb1 = torch.from_numpy(embeddings1)
        emb2 = torch.from_numpy(embeddings2)
        
        # Normalize on CPU to reduce GPU memory transfer
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        
        # Move to device after normalization
        emb1 = emb1.to(self.device)
        emb2 = emb2.to(self.device)
        
        try:
            with torch.inference_mode():
                similarities = util.cos_sim(emb1, emb2)
                result = similarities.diagonal().cpu().numpy()
                return result
        finally:
            self._cleanup_gpu_tensors([emb1, emb2, similarities])

    def calculate_pairwise_cosine_similarity(
        self,
        embeddings1: NDArray[np.float32],
        embeddings2: NDArray[np.float32],
        batch_size: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Calculate pairwise cosine similarity with memory-efficient batching."""
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {embeddings1.shape[1]} != {embeddings2.shape[1]}"
            )
        
        # Use instance batch size if none provided
        batch_size = batch_size or self.batch_size
        
        # Pre-allocate output array
        n_samples1, n_samples2 = len(embeddings1), len(embeddings2)
        output = np.zeros((n_samples1, n_samples2), dtype=np.float32)
        
        # Convert and normalize embeddings2 once (it's used for all batches)
        emb2 = torch.from_numpy(embeddings2)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        emb2 = emb2.to(self.device)
        
        try:
            for i in range(0, n_samples1, batch_size):
                batch_emb1 = torch.from_numpy(embeddings1[i:i + batch_size])
                batch_emb1 = torch.nn.functional.normalize(batch_emb1, p=2, dim=1)
                batch_emb1 = batch_emb1.to(self.device)
                
                with torch.inference_mode():
                    batch_similarities = util.cos_sim(batch_emb1, emb2)
                    output[i:i + batch_size] = batch_similarities.cpu().numpy()
                
                self._cleanup_gpu_tensors([batch_emb1, batch_similarities])
                
            return output
        finally:
            self._cleanup_gpu_tensors([emb2])

    def _cleanup_gpu_tensors(self, tensors: List[torch.Tensor]) -> None:
        """Helper method to clean up GPU tensors."""
        if self.device.type == 'cuda':
            for tensor in tensors:
                if tensor is not None and tensor.is_cuda:
                    del tensor
            torch.cuda.empty_cache()

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()

    def __len__(self) -> int:
        """Return the embedding dimension of the model."""
        return self.embedding_dimension

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory cache if using CUDA device."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def quantize(self, quantization_type: str = 'fp16') -> None:
        """Quantize model for production deployment
        
        Args:
            quantization_type: Type of quantization ('fp16' or 'int8')
        """
        if quantization_type == 'fp16':
            self.model.half()
        elif quantization_type == 'int8':
            from torch.quantization import quantize_dynamic
            self.model = quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")