from sklearn.kernel_approximation import Nystroem

from.utils import estimate_gamma_nystroem
from .svm_kernel import SVMKernel
from ..transformer import DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionDistribution,
)


class NystroemRBF(SVMKernel):
    _component_class = Nystroem
    _default_parameters = {
        "kernel": "rbf",
        "gamma": 0.1,
        "coef0": 0,
        "degree": 3,
        "kernel_params": None,
        "n_components": 500,
        "random_state": None,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.UNCOMMON

    _default_tuning_grid = {
        "gamma": FunctionDistribution(estimate_gamma_nystroem)
    }


class NystroemSigmoid(SVMKernel):
    _component_class = Nystroem
    _default_parameters = {
        "kernel": "sigmoid",
        "gamma": 0.1,
        "coef0": 0,
        "degree": 3,
        "kernel_params": None,
        "n_components": 500,
        "random_state": None,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": FunctionDistribution(estimate_gamma_nystroem),
        #"coef0": UniformDistribution(-1, 1),
    }
