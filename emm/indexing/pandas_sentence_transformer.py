from typing import Any, Callable, Optional, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.base import TransformerMixin
import torch

from emm.indexing.base_indexer import CosSimBaseIndexer
from emm.helper.blocking_functions import _parse_blocking_func
from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
from emm.models.sentence_transformer.utils import check_sentence_transformers_available

class PandasSentenceTransformerIndexer(TransformerMixin, CosSimBaseIndexer, BaseSentenceTransformerComponent):
    """Indexer using lightweight sentence transformers for initial candidate selection"""
    
    def __init__(
        self,
        input_col: str = "preprocessed",
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.5,
        num_candidates: int = 10,
        **kwargs,
    ) -> None:
        """Initialize sentence transformer indexer
        
        Args:
            input_col: Input column name containing text to encode
            model_name: Name of pre-trained model or path to fine-tuned model
                Examples:
                    - "all-MiniLM-L6-v2"  # Pre-trained
                    - "mixedbread-ai/mxbai-embed-xsmall-v1"  # With truncate_dim in model_kwargs
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
            model_kwargs: Additional kwargs for model initialization (e.g. {'truncate_dim': 384})
            encode_kwargs: Additional kwargs for encoding method
            similarity_threshold: Similarity threshold for filtering matches
            num_candidates: Number of nearest neighbors to return
            **kwargs: Additional indexer parameters
        """
        check_sentence_transformers_available()
        BaseSentenceTransformerComponent.__init__(
            self,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        CosSimBaseIndexer.__init__(self, num_candidates=num_candidates)
        self.input_col = input_col
        self.blocking_func = _parse_blocking_func(kwargs.get('blocking_func'))
        self.nn = NearestNeighbors(
            n_neighbors=num_candidates, 
            metric='cosine'
        )
        self.base_embeddings = None
        self.base_indices = None
        self.gt = None
        logger.info(f"Initializing SentenceTransformerIndexer with model {model_name}")
        self.carry_on_cols = []
        self.similarity_threshold = similarity_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> TransformerMixin:
        """Compute embeddings for base names and fit nearest neighbors
        
        Args:
            X: DataFrame containing names to encode
            y: Ignored
            
        Returns:
            self
        """
        with Timer("SentenceTransformerIndexer.fit") as timer:
            logger.info(f"Fitting indexer on {len(X)} records")
            
            if self.input_col not in X.columns:
                msg = f"Input column {self.input_col} not found in dataframe"
                raise ValueError(msg)
                
            self.gt = X  # Store ground truth like other indexers
            
            # Handle blocking if specified
            if self.blocking_func is not None:
                blocks = X[self.input_col].map(self.blocking_func)
                self.base_embeddings = {}
                self.base_indices = {}
                
                for block in blocks.unique():
                    mask = blocks == block
                    block_embeddings = self.encode_texts(
                        X[mask][self.input_col].tolist(),
                    )
                    
                    self.base_embeddings[block] = block_embeddings
                    self.base_indices[block] = X[mask].index.values
                    
                    # Fit NN per block
                    self.nn.fit(block_embeddings)
            else:
                self.base_embeddings = self.encode_texts(
                    X[self.input_col].tolist(),
                )
                
                self.base_indices = X.index.values
                self.nn.fit(self.base_embeddings)
            
            logger.info(f"Fitted indexer with {len(self.base_indices)} base vectors")
            return self

    def transform(self, X: pd.DataFrame, multiple_indexers: bool = False) -> pd.DataFrame:
        """Find nearest neighbors for query names"""
        if self.gt is None:
            msg = "Model is not fitted yet"
            raise ValueError(msg)
            
        try:
            with Timer("SentenceTransformerIndexer.transform") as timer:
                logger.info(f"Transforming {len(X)} records")
                
                # Pre-allocate results list with estimated size
                est_size = len(X) * self.num_candidates
                results = []
                results.reserve(est_size)  # Pre-allocate memory
                
                if self.blocking_func is not None:
                    # Process in blocks to reduce memory usage
                    blocks = X[self.input_col].map(self.blocking_func)
                    
                    for block in blocks.unique():
                        if block not in self.base_embeddings:
                            continue
                            
                        block_mask = blocks == block
                        block_texts = X[block_mask][self.input_col].tolist()
                        
                        # Use efficient encoding with mixed precision
                        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                            block_embeddings = self.encode_texts(block_texts)
                        
                        # Calculate similarities efficiently
                        similarities = self.calculate_pairwise_cosine_similarity(
                            block_embeddings,
                            self.base_embeddings[block],
                            batch_size=self.batch_size
                        )
                        
                        # Get top k efficiently using numpy
                        top_k_indices = np.argpartition(-similarities, 
                                                      self.num_candidates-1, 
                                                      axis=1)[:,:self.num_candidates]
                        top_k_similarities = np.take_along_axis(similarities, top_k_indices, axis=1)
                        
                        # Process matches efficiently
                        block_results = self._process_matches(
                            top_k_similarities,
                            top_k_indices,
                            X[block_mask].index,
                            self.base_indices[block]
                        )
                        results.extend(block_results)
                        
                        # Clean up memory
                        del block_embeddings, similarities, top_k_indices, top_k_similarities
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                else:
                    # Process all records at once with batching
                    query_embeddings = self.encode_texts(
                        X[self.input_col].tolist(),
                    )
                    
                    # Use efficient batch-wise similarity calculation
                    similarities = self.calculate_pairwise_cosine_similarity(
                        query_embeddings,
                        self.base_embeddings,
                        batch_size=self.batch_size
                    )
                    
                    # Get top k efficiently
                    top_k_indices = np.argpartition(-similarities, 
                                                  self.num_candidates-1, 
                                                  axis=1)[:,:self.num_candidates]
                    top_k_similarities = np.take_along_axis(similarities, top_k_indices, axis=1)
                    
                    results = self._process_matches(
                        top_k_similarities,
                        top_k_indices,
                        X.index,
                        self.base_indices
                    )
                    
                    # Clean up memory
                    del query_embeddings, similarities, top_k_indices, top_k_similarities
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Create DataFrame efficiently
                candidates = pd.DataFrame(results)
                if len(candidates) == 0:
                    candidates = pd.DataFrame(columns=['uid', 'gt_uid', 'score', 'rank'])
                
                # Optimize sorting and ranking
                candidates = candidates.sort_values(['uid', 'score'], ascending=[True, False])
                candidates['rank'] = candidates.groupby('uid').cumcount() + 1
                
                if multiple_indexers:
                    candidates[self.column_prefix()] = 1
                
                # Efficient column operations
                if self.carry_on_cols:
                    candidates = candidates.merge(
                        X[['uid'] + self.carry_on_cols], 
                        on='uid',
                        how='left'
                    )
                
                # Rename columns efficiently
                candidates = candidates.rename(columns={
                    'score': f'score_{self.column_prefix()}',
                    'rank': f'rank_{self.column_prefix()}'
                })
                
                # Filter by threshold
                candidates = candidates[candidates[f"score_{self.column_prefix()}"] >= self.similarity_threshold]
                
                logger.info(f"Generated {len(candidates)} candidates")
                return candidates

        finally:
            # Ensure memory cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def _process_matches(self, similarities, indices, query_indices, base_indices):
        """Process matches and filter by similarity threshold"""
        results = []
        mask = similarities >= self.similarity_threshold
        
        for i, (sim_row, idx_row, mask_row) in enumerate(zip(similarities, indices, mask)):
            valid_indices = idx_row[mask_row]
            valid_scores = sim_row[mask_row]
            
            for gt_idx, score in zip(valid_indices, valid_scores):
                results.append({
                    'uid': query_indices[i],
                    'gt_uid': base_indices[gt_idx],
                    'score': float(score)
                })
        return results

    def calc_score(self, name1: pd.Series, name2: pd.Series) -> pd.DataFrame:
        """Calculate similarity scores between two name series
        
        Required for compatibility with supervised model interface
        """
        assert all(name1.index == name2.index)
        
        embeddings1 = self.encode_texts(
            name1.tolist(),
        )
        
        embeddings2 = self.encode_texts(
            name2.tolist(),
        )
        
        # Calculate cosine similarities
        similarities = self.calculate_cosine_similarity(embeddings1, embeddings2)
        
        return pd.DataFrame({'score': similarities}, index=name1.index)

    def column_prefix(self) -> str:
        """Return prefix for columns created by this indexer"""
        return "sbert"