import numpy as np
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasMinMaxScaler(PandasDataFrameTransformerMixin, _MinMaxScaler):
    def get_dtypes(self, Xt, X, y=None):
        return np.float32
