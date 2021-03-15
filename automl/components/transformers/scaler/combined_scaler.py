import numpy as np
from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...flow._column_transformer import PandasColumnTransformer, make_column_selector
from .quantile_transformer import QuantileTransformer
from .standard_scaler import StandardScaler
from .robust_scaler import RobustScaler

from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.linear_model_estimator import LinearModelEstimator


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
                RobustScaler()(),
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

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, LinearModelEstimator)
        )
