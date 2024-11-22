# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Dict
import torch
from sklearn.base import TransformerMixin
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
import numpy as np

from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.supervised_model.base_supervised_model import BaseSupervisedModel
from emm.models.sentence_transformer.base import BaseSentenceTransformerComponent
from emm.models.sentence_transformer.utils import check_sentence_transformers_available

if TYPE_CHECKING:
    import pandas as pd


class SentenceTransformerLayerTransformer(TransformerMixin, BaseSupervisedModel, BaseSentenceTransformerComponent):
    """Sentence Transformer implementation for name matching"""

    def __init__(
        self,
        score_col: str = "nm_score",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize sentence transformer matcher
        
        Args:
            score_col: Name of column to store similarity scores
            model_name: Name of pre-trained model or path to fine-tuned model
                Examples:
                    - "all-MiniLM-L6-v2"  # Pre-trained
                    - "path/to/fine_tuned/scorer_model"  # Fine-tuned
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
            model_kwargs: Additional kwargs for model initialization
            encode_kwargs: Additional kwargs for encoding method
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
        BaseSupervisedModel.__init__(self)
        self.score_col = score_col
        self.similarity_threshold = similarity_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SentenceTransformerLayerTransformer:
        """Placeholder for fit method - not used as we use pre-trained models

        Args:
            X: input dataframe
            y: ignored

        Returns:
            self
        """
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Placeholder for fit_transform method

        Args:
            X: input dataframe
            y: ignored
        """
        self.fit(X, y)

    def calc_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate similarity scores using the sentence transformer model."""
        logger.info(f"Calculating similarity scores using {self.model_name}")
        
        i_to_score = X["gt_uid"].notna()
        if i_to_score.sum() == 0:
            X[self.score_col] = 0.0
            return X

        df_to_score = X[i_to_score]
        
        try:
            # Use mixed precision for encoding
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                # Calculate embeddings efficiently in batches
                name1_embeddings = self.encode_texts(df_to_score["name"].tolist())
                name2_embeddings = self.encode_texts(df_to_score["gt_name"].tolist())

            # Use optimized cosine similarity calculation
            scores = self.calculate_cosine_similarity(name1_embeddings, name2_embeddings)
            
            # Efficient assignment back to DataFrame
            X.loc[i_to_score, self.score_col] = scores

            return X
        finally:
            # Ensure cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def select_best_score(
        self,
        X: pd.DataFrame,
        group_cols: list[str],
        sort_cols: list[str] | None = None,
        sort_asc: list[bool] | None = None,
        best_match_col: str = "best_match",
        best_rank_col: str = "best_rank",
        gt_uid_col: str | None = "gt_uid",
    ) -> pd.DataFrame:
        """Select final best score from similarity scores.

        Args:
            X: pandas DataFrame with similarity scores
            group_cols: column name or list of column names used in aggregation
            sort_cols: (optional) list of columns used in ordering the results
            sort_asc: (optional) list of booleans to determine ascending order of sort_cols
            best_match_col: column indicating best match of all scores
            best_rank_col: column with rank of sorted scores
            gt_uid_col: column indicating name of gt uid
        """
        if sort_cols is None:
            sort_cols = [self.score_col]
            sort_asc = [False]
        full_sort_by = group_cols + sort_cols
        assert sort_asc is not None
        full_sort_asc = [True] * len(group_cols) + sort_asc

        # Rank candidates based on similarity scores
        X = X.sort_values(by=[*group_cols, self.score_col, gt_uid_col], ascending=False, na_position="last")
        gb = X.groupby(group_cols)
        X[best_rank_col] = gb[self.score_col].transform(lambda x: range(1, len(x) + 1))

        # Mark best matches
        X[best_match_col] = (X[best_rank_col] == 1) & (X[self.score_col].notnull()) & (X[self.score_col] > 0)

        return X.sort_values(by=full_sort_by, ascending=full_sort_asc)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Transform name pairs using sentence transformer similarity scoring.

        Args:
            X: input name-pair candidates for scoring

        Returns:
            candidates dataframe including the similarity scoring column
        """
        if X is None:
            return None

        with Timer("SentenceTransformerLayerTransformer.transform") as timer:
            timer.log_params({
                "X.shape": X.shape,
                "model": self.model_name,
                "device": self.device
            })
            
            X = self.calc_score(X)
            X = X[X[self.score_col] >= self.similarity_threshold]
            X = self.select_best_score(X, group_cols=["uid"])

            timer.log_param("cands", len(X))
        return X 

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate probability scores for name pairs."""
        # Efficient mask calculation
        valid_mask = X["name"].notna() & X["gt_name"].notna()
        n_samples = len(X)
        
        if not valid_mask.any():
            return np.zeros((n_samples, 2))
        
        try:
            # Use mixed precision for encoding
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                # Calculate embeddings efficiently in batches
                name1_embeddings = self.encode_texts(X.loc[valid_mask, "name"].tolist())
                name2_embeddings = self.encode_texts(X.loc[valid_mask, "gt_name"].tolist())

            # Calculate similarities efficiently
            similarities = util.cos_sim(
                torch.from_numpy(name1_embeddings).to(self.device),
                torch.from_numpy(name2_embeddings).to(self.device)
            )
            scores = similarities.diagonal().cpu().numpy()
            
            # Efficient array operations
            full_scores = np.zeros(n_samples)
            full_scores[valid_mask] = scores
            
            return np.vstack([(1 - full_scores), full_scores]).T
        finally:
            # Clean up GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()