from sklearn.preprocessing import RobustScaler as _RobustScaler

from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

class PandasRobustScaler(PandasDataFrameTransformerMixin, _RobustScaler):
    pass

class RobustScaler(Scaler):
    _component_class = PandasRobustScaler
    _default_parameters = {
        "with_centering": True,
        "with_scaling" : True,
        "quantile_range": (25.0, 75.0),
        "copy": True,
        "unit_variance": False,
    }
    _allowed_dtypes = {
        DataType.NUMERIC
    }
    _component_level = ComponentLevel.NECESSARY