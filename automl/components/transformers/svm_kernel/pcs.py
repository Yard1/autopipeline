from .svm_kernel import SVMKernel
from ..transformer import DataType
from ...component import ComponentLevel
from ....search.distributions import (
    CategoricalDistribution,
    IntUniformDistribution,
)
from ...component import ComponentConfig
from ....search.stage import AutoMLStage

from automl_models.components.transformers.svm_kernel.pcs import (
    PolynomialCountSketchDynamicNComponents,
)


class PolynomialCountSketch(SVMKernel):
    _component_class = PolynomialCountSketchDynamicNComponents
    _default_parameters = {
        "gamma": "scale",
        "coef0": 0,
        "degree": 3,
        "n_components": None,
        "random_state": 0,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", "auto"]),
        #    "coef0": UniformDistribution(-1, 1),
        "degree": IntUniformDistribution(2, 3),
    }
