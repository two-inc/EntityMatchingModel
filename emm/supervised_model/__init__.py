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

from emm.helper import spark_installed
from emm.supervised_model.base_supervised_model import train_model, train_test_model
from emm.supervised_model.pandas_supervised_model import PandasSupervisedLayerTransformer
from emm.supervised_model.sentence_transformer_model import SentenceTransformerLayerTransformer

__all__ = [
    "train_model", 
    "train_test_model", 
    "PandasSupervisedLayerTransformer",
    "SentenceTransformerLayerTransformer"
]

if spark_installed:
    from emm.supervised_model.spark_supervised_model import SparkSupervisedLayerEstimator
    __all__ += ["SparkSupervisedLayerEstimator"]
