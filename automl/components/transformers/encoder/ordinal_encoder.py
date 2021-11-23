import numpy as np
from .encoder import Encoder
from ..transformer import DataType
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage

from automl_models.components.transformers.encoder.ordinal_encoder import (
    PandasOrdinalEncoder,
)


class BinaryEncoder(Encoder):
    _component_class = PandasOrdinalEncoder
    _default_parameters = {
        "categories": "auto",
        "handle_unknown": "error",
        "dtype": bool,
        "unknown_value": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )

class OrdinalEncoder(Encoder):
    _component_class = PandasOrdinalEncoder
    _default_parameters = {
        "categories": "auto",
        "handle_unknown": "error",
        "dtype": np.uint16,
        "unknown_value": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or getattr(config.estimator, "_has_own_cat_encoding", False)
        )
