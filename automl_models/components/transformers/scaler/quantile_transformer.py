from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasQuantileTransformer(PandasDataFrameTransformerMixin, _QuantileTransformer):
    pass
