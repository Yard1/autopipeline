from .svm_kernel import SVMKernel
from ..transformer import DataType
from ...component import ComponentLevel
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
        "n_components": None,
        "random_state": 0,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", "auto"])
    }


class NystroemSigmoid(SVMKernel):
    _component_class = NystroemDynamicGamma
    _default_parameters = {
        "kernel": "sigmoid",
        "gamma": "scale",
        "coef0": 0,
        "degree": 3,
        "kernel_params": None,
        "n_components": None,
        "random_state": 0,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", "auto"]),
        # "coef0": UniformDistribution(-1, 1),
    }
