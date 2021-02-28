import numpy as np
from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...flow._column_transformer import PandasColumnTransformer
from .quantile_transformer import QuantileTransformer
from .standard_scaler import StandardScaler

from ....search.blueprints.column_selector import make_column_selector


def _scaler_skewness_condition(column, skewness_threshold=0.99):
    return np.abs(column.skew()) > skewness_threshold


class PandasCombinedScalerTransformer(PandasColumnTransformer):
    def __init__(
        self,
        *,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False
    ):
        self.transformers = [
            (
                "Scaler",
                StandardScaler()(),
                make_column_selector(
                    _scaler_skewness_condition,
                    negate_condition=True,
                ),
            ),
            (
                "Transformer",
                QuantileTransformer()(),
                make_column_selector(
                    _scaler_skewness_condition,
                    negate_condition=False,
                ),
            ),
        ]
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose


class CombinedScalerTransformer(Scaler):
    _component_class = PandasCombinedScalerTransformer
    _default_parameters = {
        "sparse_threshold": 0.3,
        "n_jobs": None,
        "transformer_weights": None,
        "verbose": False,
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY