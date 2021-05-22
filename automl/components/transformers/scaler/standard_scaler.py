from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel

from automl_models.components.transformers.scaler.standard_scaler import (
    PandasStandardScaler,
)


class StandardScaler(Scaler):
    _component_class = PandasStandardScaler
    _default_parameters = {
        "copy": True,
        "with_mean": True,
        "with_std": True,
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY
