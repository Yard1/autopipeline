from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.linear_model_estimator import LinearModelEstimator

from automl_models.components.transformers.scaler.combined_scaler import (
    PandasCombinedScalerTransformer,
)


class CombinedScalerTransformer(Scaler):
    _component_class = PandasCombinedScalerTransformer
    _default_parameters = {
        "sparse_threshold": 0.3,
        "n_jobs": 1,
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
            config.estimator is None
            or isinstance(config.estimator, LinearModelEstimator)
        )
