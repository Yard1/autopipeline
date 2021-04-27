from .svm_kernel import SVMKernel
from ..transformer import DataType
from ...component import ComponentLevel
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.svm.svm import SVM
from ....search.distributions import (
    CategoricalDistribution,
)


from automl_models.components.transformers.svm_kernel.nystroem import (
    NystroemDynamicGamma,
)


class NystroemRBF(SVMKernel):
    _component_class = NystroemDynamicGamma
    _default_parameters = {
        "kernel": "rbf",
        "gamma": "scale",
        "coef0": 0,
        "degree": 3,
        "kernel_params": None,
        "n_components": 500,
        "random_state": 0,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", 1.0, 0.1, "auto"])
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, SVM)
        )


class NystroemSigmoid(SVMKernel):
    _component_class = NystroemDynamicGamma
    _default_parameters = {
        "kernel": "sigmoid",
        "gamma": "scale",
        "coef0": 0,
        "degree": 3,
        "kernel_params": None,
        "n_components": 500,
        "random_state": 0,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", 1.0, 0.1, "auto"]),
        # "coef0": UniformDistribution(-1, 1),
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, SVM)
        )
