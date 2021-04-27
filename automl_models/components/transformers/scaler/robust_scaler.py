from sklearn.preprocessing import RobustScaler as _RobustScaler

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasRobustScaler(PandasDataFrameTransformerMixin, _RobustScaler):
    pass
