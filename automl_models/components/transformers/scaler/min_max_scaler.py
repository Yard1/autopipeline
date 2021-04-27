from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasMinMaxScaler(PandasDataFrameTransformerMixin, _MinMaxScaler):
    pass
