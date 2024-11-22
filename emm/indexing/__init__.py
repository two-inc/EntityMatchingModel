# Copyright (c) 2023 ING Analytics Wholesale Banking
#
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

"""Entity matching indexers for candidate pair generation.

Core indexers:
- PandasCosSimIndexer: Cosine similarity based indexing
- PandasNaiveIndexer: Simple O(n^2) indexing for small datasets
- PandasSortedNeighbourhoodIndexer: Sorted neighborhood indexing

Optional indexers (require additional dependencies):
- Spark indexers (requires pyspark):
  - SparkCosSimIndexer
  - SparkCandidateSelectionEstimator
  - SparkSortedNeighbourhoodIndexer
- Sentence Transformer indexers (requires sentence-transformers):
  - PandasSentenceTransformerIndexer
"""

from __future__ import annotations

from emm.indexing.pandas_cos_sim_matcher import PandasCosSimIndexer
from emm.indexing.pandas_naive_indexer import PandasNaiveIndexer
from emm.indexing.pandas_sni import PandasSortedNeighbourhoodIndexer

__all__ = [
    "PandasCosSimIndexer", 
    "PandasNaiveIndexer", 
    "PandasSortedNeighbourhoodIndexer",
]

# Optional Spark support
try:
    import pyspark
    from emm.indexing.spark_cos_sim_matcher import SparkCosSimIndexer
    from emm.indexing.spark_candidate_selection import SparkCandidateSelectionEstimator
    from emm.indexing.spark_sni import SparkSortedNeighbourhoodIndexer
    __all__.extend([
        "SparkCosSimIndexer",
        "SparkCandidateSelectionEstimator", 
        "SparkSortedNeighbourhoodIndexer"
    ])
except ImportError:
    pass

# Optional Sentence Transformer support 
try:
    from emm.indexing.pandas_sentence_transformer import PandasSentenceTransformerIndexer
    __all__.extend([
        "PandasSentenceTransformerIndexer"
    ])
except ImportError:
    pass  # Transformer features unavailable
