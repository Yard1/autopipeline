from sklearn.impute import SimpleImputer as _SimpleImputer

from ..transformer import Transformer, DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ....search.distributions import CategoricalDistribution

class PandasSimpleImputer(PandasDataFrameTransformerMixin, _SimpleImputer):
    pass

class SimpleNumericImputer(Transformer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "mean",
        "fill_value": 0,
        "verbose": 0,
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": CategoricalDistribution(["mean", "median", "constant"])
    }
    _allowed_dtypes = {
        DataType.NUMERIC
    }
    _component_level = ComponentLevel.NECESSARY

class SimpleCategoricalImputer(Transformer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "most_frequent",
        "fill_value": "missing_value",
        "verbose": 0,
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": CategoricalDistribution(["most_frequent", "constant"])
    }
    _allowed_dtypes = {
        DataType.CATEGORICAL
    }
    _component_level = ComponentLevel.NECESSARY