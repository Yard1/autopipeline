from sklearn.preprocessing import StandardScaler as _StandardScaler

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasStandardScaler(PandasDataFrameTransformerMixin, _StandardScaler):
    pass
