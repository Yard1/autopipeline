from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel, ProblemType, ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.linear_model_estimator import LinearModelEstimator
from ...estimators.neural_network.neural_network_estimator import NeuralNetworkEstimator
from automl_models.components.transformers.scaler.quantile_transformer import (
    PandasQuantileTransformer,
)


class QuantileTransformer(Scaler):
    _component_class = PandasQuantileTransformer
    _default_parameters = {
        "n_quantiles": 1000,
        "output_distribution": "normal",
        "ignore_implicit_zeros": False,
        "subsample": 1e5,
        "random_state": 0,
        "copy": True,
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY


class QuantileTargetTransformer(Scaler):
    _component_class = PandasQuantileTransformer
    _default_parameters = {
        "n_quantiles": 1000,
        "output_distribution": "normal",
        "ignore_implicit_zeros": False,
        "subsample": 1e5,
        "random_state": 0,
        "copy": True,
    }
    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = ComponentLevel.UNCOMMON

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or isinstance(config.estimator, (LinearModelEstimator, NeuralNetworkEstimator))
        )
