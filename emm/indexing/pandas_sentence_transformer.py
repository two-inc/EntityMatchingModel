from typing import Any, Callable, Optional, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.base import TransformerMixin

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
        CosSimBaseIndexer.__init__(self, **kwargs)
        self.input_col = input_col
        self.blocking_func = _parse_blocking_func(kwargs.get('blocking_func'))
        self.nn = NearestNeighbors(
            n_neighbors=kwargs.get('num_candidates', 10), 
            metric='cosine'
        )
        self.base_embeddings = None
        self.base_indices = None
        self.gt = None
        logger.info(f"Initializing SentenceTransformerIndexer with model {model_name}")
        self.carry_on_cols = []

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
                
                results = []
                
                if self.blocking_func is not None:
                    blocks = X[self.input_col].map(self.blocking_func)
                    for block in blocks.unique():
                        if block not in self.base_embeddings:
                            continue
                            
                        block_embeddings = self.encode_texts(
                            X[blocks == block][self.input_col].tolist(),
                        )
                        
                        distances, indices = self.nn.kneighbors(block_embeddings)
                        similarities = 1 - distances
                        
                        block_results = self._process_matches(
                            similarities,
                            indices,
                            X[blocks == block].index,
                            self.base_indices[block]
                        )
                        results.extend(block_results)
                        
                        # Clean up memory after each block
                        del block_embeddings
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                else:
                    query_embeddings = self.encode_texts(
                        X[self.input_col].tolist(),
                    )
                    
                    distances, indices = self.nn.kneighbors(query_embeddings)
                    similarities = 1 - distances
                    
                    results = self._process_matches(
                        similarities,
                        indices,
                        X.index,
                        self.base_indices
                    )
                    
                    # Clean up memory
                    del query_embeddings
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                candidates = pd.DataFrame(results)
                if len(candidates) == 0:
                    candidates = pd.DataFrame(columns=['uid', 'gt_uid', 'score', 'rank'])
                    
                # Sort and rank within groups like other indexers
                candidates = candidates.sort_values(['uid', 'score'], ascending=[True, False])
                gb = candidates.groupby('uid')
                candidates['rank'] = gb['score'].transform(lambda x: range(1, len(x) + 1))
                
                if multiple_indexers:
                    candidates[self.column_prefix()] = 1
                    
                if self.carry_on_cols:
                    for col in self.carry_on_cols:
                        if col in X.columns:
                            candidates[col] = X[col]
                            
                candidates = candidates.rename(columns={
                    'score': f'score_{self.column_prefix()}',
                    'rank': f'rank_{self.column_prefix()}'
                })
                
                logger.info(f"Generated {len(candidates)} candidates")
                return candidates

        finally:
            # Ensure memory is cleaned up even if an error occurs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def _process_matches(self, similarities, indices, query_indices, base_indices):
        """Process matches and filter by similarity threshold"""
        results = []
        mask = similarities >= self.cos_sim_lower_bound
        
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