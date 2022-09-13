from .imputer import Imputer
from ..transformer import DataType
from ...component import ComponentLevel
from ....search.distributions import CategoricalDistribution

from automl_models.components.transformers.imputer.simple_imputer import (
    PandasSimpleCategoricalImputer,
    PandasSimpleImputer,
)


class SimpleNumericImputer(Imputer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "median",
        "fill_value": 0,
        "verbose": "deprecated",
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {"strategy": CategoricalDistribution(["mean", "median"])}
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY


class SimpleCategoricalImputer(Imputer):
    _component_class = PandasSimpleCategoricalImputer
    _default_parameters = {
        "strategy": "most_frequent",
        "fill_value": "missing_value",
        "verbose": "deprecated",
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": CategoricalDistribution(["most_frequent", "constant"])
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY
