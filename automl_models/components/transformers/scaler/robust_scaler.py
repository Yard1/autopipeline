import numpy as np
from sklearn.preprocessing import RobustScaler as _RobustScaler

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasRobustScaler(PandasDataFrameTransformerMixin, _RobustScaler):
    def get_dtypes(self, Xt, X, y=None):
        return np.float32
