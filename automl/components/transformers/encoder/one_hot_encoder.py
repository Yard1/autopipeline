import numpy as np
from .encoder import Encoder
from ..transformer import DataType
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.tree.tree_estimator import TreeEstimator

from automl_models.components.transformers.encoder.one_hot_encoder import (
    PandasOneHotEncoder,
)


class OneHotEncoder(Encoder):
    _component_class = PandasOneHotEncoder
    _default_parameters = {
        "categories": "auto",
        "drop": "if_binary",
        "sparse": False,
        "dtype": np.bool,
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
            and (
                config.X is None
                or sum(
                    [
                        len(config.X[col].cat.categories)
                        if len(config.X[col].cat.categories) > 2
                        else 0
                        for col in config.X.select_dtypes("category")
                    ]
                )
                <= config.X.shape[1] * 2
            )
        )
