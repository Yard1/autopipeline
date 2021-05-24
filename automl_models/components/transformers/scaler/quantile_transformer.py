import numpy as np
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasQuantileTransformer(PandasDataFrameTransformerMixin, _QuantileTransformer):
    def get_dtypes(self, Xt, X, y=None):
        return np.float32
