from sklearn.preprocessing import StandardScaler as _StandardScaler

from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

class PandasStandardScaler(PandasDataFrameTransformerMixin, _StandardScaler):
    pass

class StandardScaler(Scaler):
    _component_class = PandasStandardScaler
    _default_parameters = {
        "copy": True,
        "with_mean" : True,
        "with_std": True,
    }
    _allowed_dtypes = {
        DataType.NUMERIC
    }
    _component_level = ComponentLevel.NECESSARY