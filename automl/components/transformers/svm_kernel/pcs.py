from .svm_kernel import SVMKernel
from ..transformer import DataType
from ...component import ComponentLevel
from ....search.distributions import (
    CategoricalDistribution,
    IntUniformDistribution,
)
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.linear_model_estimator import LinearModelEstimator

from automl_models.components.transformers.svm_kernel.pcs import (
    PolynomialCountSketchDynamicNComponents,
)


class PolynomialCountSketch(SVMKernel):
    _component_class = PolynomialCountSketchDynamicNComponents
    _default_parameters = {
        "gamma": 1.0,
        "coef0": 0,
        "degree": 3,
        "n_components": 500,
        "random_state": 0,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", 1.0, "auto"]),
        #    "coef0": UniformDistribution(-1, 1),
        "degree": IntUniformDistribution(2, 3),
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or isinstance(config.estimator, LinearModelEstimator)
        )
