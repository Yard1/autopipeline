import numpy as np
from sklearn.preprocessing import StandardScaler as _StandardScaler

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasStandardScaler(PandasDataFrameTransformerMixin, _StandardScaler):
    def get_dtypes(self, Xt, X, y=None):
        return np.float32
