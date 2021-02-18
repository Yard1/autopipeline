from sklearn.impute import SimpleImputer as _SimpleImputer

from .transformer import Transformer, DataType
from ..compatibility.pandas import PandasDataFrameTransformerMixin

class PandasSimpleImputer(PandasDataFrameTransformerMixin, _SimpleImputer):
    pass

class SimpleNumericalImputer(Transformer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "mean",
        "fill_value": 0,
        "verbose": 0,
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": ["mean", "median", "constant"]
    }
    _allowed_dtypes = {
        DataType.NUMERICAL
    }

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
        "strategy": ["most_frequent", "constant"]
    }
    _allowed_dtypes = {
        DataType.CATEGORICAL
    }