from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel

from automl_models.components.transformers.scaler.quantile_transformer import (
    PandasQuantileTransformer,
)


class QuantileTransformer(Scaler):
    _component_class = PandasQuantileTransformer
    _default_parameters = {
        "n_quantiles": 1000,
        "output_distribution": "normal",
        "ignore_implicit_zeros": False,
        "subsample": 1e5,
        "random_state": 0,
        "copy": True,
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY
