from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.knn.knn_estimator import KNNEstimator

from automl_models.components.transformers.scaler.min_max_scaler import (
    PandasMinMaxScaler,
)


class MinMaxScaler(Scaler):
    _component_class = PandasMinMaxScaler
    _default_parameters = {
        "feature_range": (0, 1),
        "copy": True,
        "clip": False,
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, KNNEstimator)
        )
