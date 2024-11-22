"""Base sentence transformer component for entity matching.

This module requires sentence-transformers and torch.
Install with: pip install emm[transformers]
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, Tuple, Union
import logging
import numpy as np

# Single numpy typing import with fallback
try:
    from numpy.typing import NDArray
except ImportError:
    from typing import Any
    NDArray = Any

logger = logging.getLogger(__name__)

class BaseSentenceTransformerComponent:
    """Base component for sentence transformer functionality in EMM."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        device: Optional[str] = None,
        batch_size: int = 32,
        use_fp16: Optional[bool] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the base sentence transformer component.
        
        Args:
            model_name: Name or path of the sentence transformer model
            similarity_threshold: Threshold for similarity matching
            device: Device to run model on ('cuda', 'cpu', etc)
            batch_size: Batch size for encoding
            use_fp16: Whether to use FP16. If None, automatically enables for CUDA devices
            model_kwargs: Additional kwargs passed to SentenceTransformer initialization
            encode_kwargs: Additional kwargs passed to encode() method
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers and torch are required. "
                "Install with: pip install emm[transformers]"
            ) from e
            
        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
            
        # FP16 setup - enable by default for CUDA if not explicitly disabled
        if use_fp16 is None:
            use_fp16 = device == 'cuda'
        self.use_fp16 = use_fp16

        if self.use_fp16 and device != 'cuda':
            logger.warning(
                "FP16 requested but device is not CUDA. Disabling FP16."
            )
            self.use_fp16 = False
            
        # Model initialization with optional kwargs
        model_kwargs = model_kwargs or {}
        self.model = SentenceTransformer(model_name, device=device, **model_kwargs)
        if self.use_fp16:
            self.model.half()  # Convert model to FP16
            
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        
        # Store encode kwargs for use in encode_texts
        self.encode_kwargs = {
            'batch_size': self.batch_size,
            'normalize_embeddings': True,
            'convert_to_tensor': True,
            'show_progress_bar': False,
        }
        # Update with user-provided encode kwargs
        if encode_kwargs:
            self.encode_kwargs.update(encode_kwargs)

    def encode_texts(self, texts: List[str], return_tensor: bool = False) -> Union[torch.Tensor, NDArray[np.float32]]:
        """Encode texts with optional FP16 precision.
        
        Args:
            texts: List of texts to encode
            return_tensor: If True, return torch.Tensor instead of numpy array
                (useful for intermediate computations)
            
        Returns:
            Embeddings as either torch.Tensor (FP16 if enabled) or numpy array (FP32)
        """
        if self.use_fp16:
            import torch
            with torch.cuda.amp.autocast():
                with torch.inference_mode():
                    embeddings = self.model.encode(texts, **self.encode_kwargs)
        else:
            with torch.inference_mode():
                embeddings = self.model.encode(texts, **self.encode_kwargs)
        
        # Keep as tensor for intermediate computations
        if return_tensor:
            return embeddings
        
        # Convert to FP32 numpy only for final output
        return embeddings.float().cpu().numpy()

    def calculate_similarity(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> NDArray[np.float32]:
        """Calculate similarity between two lists of texts.
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            Array of similarity scores in FP32
        """
        # Keep tensors in FP16 for intermediate computations
        embeddings1 = self.encode_texts(texts1, return_tensor=True)
        embeddings2 = self.encode_texts(texts2, return_tensor=True)
        
        # Compute similarity while maintaining precision
        similarities = self.model.similarity(embeddings1, embeddings2)
        
        # Convert to numpy only at the end
        return similarities.float().cpu().numpy()

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by the model."""
        return self.model.get_sentence_embedding_dimension()