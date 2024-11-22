from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors
from typing import Any, List, Optional

from emm.helper.blocking_functions import _parse_blocking_func
from emm.indexing.base_indexer import SentenceTransformerBaseIndexer
from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent

class PandasSentenceTransformerIndexer(TransformerMixin, SentenceTransformerBaseIndexer):
    """Indexer using lightweight sentence transformers for initial candidate selection"""
    
    def __init__(
        self,
        input_col: str = "name",
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
        similarity_threshold: float = 0.5,
        num_candidates: int = 5,
        **kwargs,
    ):
        """Initialize the sentence transformer indexer"""
        # Change from super().__init__() to explicit parent call
        SentenceTransformerBaseIndexer.__init__(
            self,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            similarity_threshold=similarity_threshold,
        )
        
        # Initialize instance attributes
        self.input_col = input_col
        self.num_candidates = num_candidates
        self.blocking_func = _parse_blocking_func(kwargs.get('blocking_func'))
        self.carry_on_cols = []
        
        # Initialize the sentence transformer component
        self.transformer = BaseSentenceTransformerComponent(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            similarity_threshold=similarity_threshold,
        )
        
        # Initialize nearest neighbors
        self.nn = NearestNeighbors(n_neighbors=num_candidates, metric='cosine')
        
        # Storage for fit data
        self.base_embeddings = None
        self.base_indices = None
        self.gt = None
        
        logger.info(f"Initialized SentenceTransformerIndexer with model {model_name}")

    def fit(self, X: pd.DataFrame, y: Any = None) -> TransformerMixin:
        """Compute embeddings for base names and fit nearest neighbors"""
        with Timer("SentenceTransformerIndexer.fit") as timer:
            timer.log_params({"n_records": len(X)})
            
            if self.input_col not in X.columns:
                raise ValueError(f"Input column {self.input_col} not found in dataframe")
                
            self.gt = X
            
            try:
                if self.blocking_func is not None:
                    self._fit_with_blocking(X)
                else:
                    self._fit_without_blocking(X)
                
                logger.info(f"Fitted indexer with {len(self.base_indices)} base vectors")
                return self
                
            finally:
                self.transformer.clear_gpu_memory()

    def _fit_with_blocking(self, X: pd.DataFrame) -> None:
        """Fit indexer with blocking"""
        blocks = X[self.input_col].map(self.blocking_func)
        self.base_embeddings = {}
        self.base_indices = {}
        
        for block in blocks.unique():
            mask = blocks == block
            block_embeddings = self.transformer.encode_texts(X[mask][self.input_col].tolist())
            self.base_embeddings[block] = block_embeddings
            self.base_indices[block] = X[mask].index.values
            self.nn.fit(block_embeddings)

    def _fit_without_blocking(self, X: pd.DataFrame) -> None:
        """Fit indexer without blocking"""
        self.base_embeddings = self.transformer.encode_texts(X[self.input_col].tolist())
        self.base_indices = X.index.values
        self.nn.fit(self.base_embeddings)

    def transform(self, X: pd.DataFrame, multiple_indexers: bool = False) -> pd.DataFrame:
        """Find nearest neighbors for query names"""
        if self.gt is None:
            raise ValueError("Model is not fitted yet")
            
        with Timer("SentenceTransformerIndexer.transform") as timer:
            timer.log_params({"n_records": len(X)})
            
            try:
                results = []
                if self.blocking_func is not None:
                    results = self._transform_with_blocking(X)
                else:
                    results = self._transform_without_blocking(X)
                
                candidates = self._format_results(results, X, multiple_indexers)
                logger.info(f"Generated {len(candidates)} candidates")
                return candidates
                
            finally:
                self.transformer.clear_gpu_memory()

    def _transform_with_blocking(self, X: pd.DataFrame) -> list:
        """Transform with blocking"""
        results = []
        blocks = X[self.input_col].map(self.blocking_func)
        
        for block in blocks.unique():
            if block not in self.base_embeddings:
                continue
                
            block_mask = blocks == block
            block_texts = X[block_mask][self.input_col].tolist()
            query_embeddings = self.transformer.encode_texts(block_texts)
            distances, indices = self.nn.kneighbors(query_embeddings)
            
            similarities = 1 - distances
            mask = similarities >= self.similarity_threshold
            
            for i, (sim_row, idx_row, mask_row) in enumerate(zip(similarities, indices, mask)):
                valid_indices = idx_row[mask_row]
                valid_scores = sim_row[mask_row]
                
                for gt_idx, score in zip(valid_indices, valid_scores):
                    results.append({
                        'uid': X[block_mask].index[i],
                        'gt_uid': self.base_indices[block][gt_idx],
                        'score': float(score)
                    })
        
        return results

    def _transform_without_blocking(self, X: pd.DataFrame) -> list:
        """Transform without blocking"""
        results = []
        query_embeddings = self.transformer.encode_texts(X[self.input_col].tolist())
        distances, indices = self.nn.kneighbors(query_embeddings)
        
        similarities = 1 - distances
        mask = similarities >= self.similarity_threshold
        
        for i, (sim_row, idx_row, mask_row) in enumerate(zip(similarities, indices, mask)):
            valid_indices = idx_row[mask_row]
            valid_scores = sim_row[mask_row]
            
            for gt_idx, score in zip(valid_indices, valid_scores):
                results.append({
                    'uid': X.index[i],
                    'gt_uid': self.base_indices[gt_idx],
                    'score': float(score)
                })
        
        return results

    def _format_results(self, results: list, X: pd.DataFrame, multiple_indexers: bool) -> pd.DataFrame:
        """Format results into output DataFrame"""
        candidates = pd.DataFrame(results)
        if len(candidates) == 0:
            candidates = pd.DataFrame(columns=['uid', 'gt_uid', 'score', 'rank'])
            
        candidates = candidates.sort_values(['uid', 'score'], ascending=[True, False])
        candidates['rank'] = candidates.groupby('uid').cumcount() + 1
        
        if multiple_indexers:
            candidates[self.column_prefix()] = 1
            
        if self.carry_on_cols:
            candidates = candidates.merge(
                X[['uid'] + self.carry_on_cols],
                on='uid',
                how='left'
            )
            
        prefix = self.column_prefix()
        candidates = candidates.rename(columns={
            'score': f'score_{prefix}',
            'rank': f'rank_{prefix}'
        })
        
        return candidates

    def calc_score(self, name1: pd.Series, name2: pd.Series) -> pd.DataFrame:
        """Calculate similarity scores between two name series"""
        assert all(name1.index == name2.index)
        
        embeddings1 = self.transformer.encode_texts(name1.tolist())
        embeddings2 = self.transformer.encode_texts(name2.tolist())
        
        similarities = self.transformer.calculate_cosine_similarity(embeddings1, embeddings2)
        return pd.DataFrame({'score': similarities}, index=name1.index)

    def column_prefix(self) -> str:
        """Return prefix for columns created by this indexer"""
        return "sbert"