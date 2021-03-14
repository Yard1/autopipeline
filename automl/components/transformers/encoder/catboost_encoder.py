import numpy as np
import pandas as pd

from category_encoders.cat_boost import CatBoostEncoder as _CatBoostEncoder

from .encoder import Encoder
from ..transformer import DataType
from ..utils import categorical_column_to_int_categories
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage


# TODO: consider K-fold inside? something like cross_val_predict
class FixedCatBoostEncoder(_CatBoostEncoder):
    def fit_transform(self, X, y):
        return super().fit(X, y).transform(X)

    def fit(self, X, y, **kwargs):
        X = X.apply(categorical_column_to_int_categories).reset_index(drop=True)
        return super().fit(X, y, **kwargs)

    def transform(self, X, y=None, override_return_df=False):
        X_index = X.index
        X = X.apply(categorical_column_to_int_categories).reset_index(drop=True)
        Xt = super().transform(X, y=y, override_return_df=override_return_df)
        Xt.index = X_index
        return Xt


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
        "sigma": None,
        "a": 1,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )
