import numpy as np
from ...flow.column_transformer import PandasColumnTransformer, make_column_selector
from .quantile_transformer import PandasQuantileTransformer
from .robust_scaler import PandasRobustScaler


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
                PandasRobustScaler(),
                make_column_selector(
                    _scaler_skewness_condition,
                    negate_condition=True,
                ),
            ),
            (
                "Transformer",
                PandasQuantileTransformer(),
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
