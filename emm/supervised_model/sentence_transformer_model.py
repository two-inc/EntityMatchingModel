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

from typing import TYPE_CHECKING, Any, Mapping
import torch
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from emm.loggers import Timer
from emm.loggers.logger import logger
from emm.supervised_model.base_supervised_model import BaseSupervisedModel

if TYPE_CHECKING:
    import pandas as pd


class SentenceTransformerLayerTransformer(TransformerMixin, BaseSupervisedModel):
    """Sentence Transformer implementation for name matching"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        score_col: str = "nm_score",
        device: str | None = None,
        batch_size: int = 32,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Sentence Transformer implementation for name matching

        SentenceTransformerLayerTransformer is used as an alternative to the supervised model transformer
        in the pipeline of PandasEntityMatching. It uses pre-trained sentence transformer models to
        compute semantic similarity between name pairs.

        Args:
            model_name: name of the pre-trained sentence transformer model to use
            score_col: name of the column to store similarity scores
            device: device to run the model on ('cpu', 'cuda', or None for auto-detection)
            batch_size: batch size for processing embeddings
            args: ignored
            kwargs: ignored

        Examples:
            >>> transformer = SentenceTransformerLayerTransformer(model_name='all-MiniLM-L6-v2')
            >>> scored_df = transformer.transform(candidates_df)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with `pip install emm[transformers]` "
                "or `pip install sentence-transformers`"
            )
            
        self.model_name = model_name
        self.score_col = score_col
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the model
        self.model = SentenceTransformer(model_name, device=self.device)
        BaseSupervisedModel.__init__(self)

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
        """Calculate similarity scores using the sentence transformer model.

        Args:
            X: DataFrame containing name pairs to score
        
        Returns:
            DataFrame with added similarity scores
        """
        logger.info(f"Calculating similarity scores using {self.model_name}")
        
        # Get name pairs to compare
        i_to_score = X["gt_uid"].notna()
        if i_to_score.sum() == 0:
            X[self.score_col] = 0.0
            return X

        # Get the relevant rows
        df_to_score = X[i_to_score]
        
        # Calculate embeddings in batches
        name1_embeddings = self.model.encode(
            df_to_score["name"].tolist(),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        name2_embeddings = self.model.encode(
            df_to_score["gt_name"].tolist(),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )

        # Calculate cosine similarities
        similarities = util.cos_sim(name1_embeddings, name2_embeddings)
        
        # Convert to numpy and flatten
        scores = similarities.diagonal().cpu().numpy()
        
        # Assign scores back to DataFrame
        X.loc[i_to_score, self.score_col] = scores

        return X

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
            X = self.select_best_score(X, group_cols=["uid"])

            timer.log_param("cands", len(X))
        return X 