import numpy as np
import pandas as pd

from category_encoders.cat_boost import CatBoostEncoder as _CatBoostEncoder

from .encoder import Encoder
from ..transformer import DataType
from ..utils import categorical_column_to_int_categories
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage


class FixedCatBoostEncoder(_CatBoostEncoder):
    def fit_transform(self, X, y):
        return super().fit(X, y).transform(X)

    def fit(self, X, y, **kwargs):
        X = X.apply(categorical_column_to_int_categories)
        return super().fit(X, y, **kwargs)

    def transform(self, X, y=None, override_return_df=False):
        X = X.apply(categorical_column_to_int_categories)
        return super().transform(X, y=y, override_return_df=override_return_df)


class CatBoostEncoder(Encoder):
    _component_class = FixedCatBoostEncoder
    _default_parameters = {
        "verbose": 0,
        "cols": None,
        "drop_invariant": False,
        "return_df": True,
        "handle_unknown": "value",
        "handle_missing": "value",
        "random_state": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.UNCOMMON

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )
