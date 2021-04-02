import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder

from .encoder import Encoder
from ..transformer import DataType
from ...component import ComponentLevel, ComponentConfig
from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ....search.stage import AutoMLStage
from ...estimators.tree.tree_estimator import TreeEstimator


class PandasOneHotEncoder(PandasDataFrameTransformerMixin, _OneHotEncoder):
    def get_columns(self, Xt, X, y=None):
        columns = []
        for column, categories in zip(self.columns_, self.categories_):
            if self.drop == "first" or (
                self.drop == "if_binary" and len(categories) == 2
            ):
                categories = categories[1:]
            columns.extend([f"{column}_{category}" for category in categories])
        return columns

    def get_dtypes(self, Xt, X, y=None):
        return pd.CategoricalDtype([0, 1])

    def _validate_keywords(self):
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', "
                "got {0}.".format(self.handle_unknown)
            )
            raise ValueError(msg)

    def fit(self, X, y=None):
        # we assume we only have categorical columns here
        X = X.apply(lambda col: col.cat.codes)
        return super().fit(X, y=y)

    def transform(self, X):
        # we assume we only have categorical columns here
        X = X.apply(lambda col: col.cat.codes)
        return super().transform(X)


class OneHotEncoder(Encoder):
    _component_class = PandasOneHotEncoder
    _default_parameters = {
        "categories": "auto",
        "drop": "if_binary",
        "sparse": False,
        "dtype": np.int,
        "handle_unknown": "ignore",
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return (
            super_check
            and (
                config.estimator is None
                or not getattr(config.estimator, "_has_own_cat_encoding", False)
            )
            and (
                config.estimator is None
                or not isinstance(config.estimator, TreeEstimator)
            )
        )
