import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.impute._base import (
    _get_mask,
    _most_frequent,
    _check_inputs_dtype,
    FLOAT_DTYPES,
    is_scalar_nan,
)
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.utils.validation import check_is_fitted

from .imputer import Imputer
from ..transformer import Transformer, DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ....search.distributions import CategoricalDistribution


class PandasSimpleImputer(PandasDataFrameTransformerMixin, _SimpleImputer):
    pass


class PandasSimpleCategoricalImputer(PandasSimpleImputer):
    def _validate_input(self, X, in_fit):
        allowed_strategies = ["most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} "
                " got strategy={1}".format(allowed_strategies, self.strategy)
            )

        if not X.dtypes.apply(is_categorical_dtype).all():
            raise ValueError(
                "Only Pandas DataFrames with all Categorical dtypes are supported."
            )

        return X

    def fit(self, X, y=None):
        X = self._validate_input(X, in_fit=True)
        fill_value = self.fill_value or "missing_value"
        if self.strategy == "constant":
            self.statistics_ = {col: fill_value for col in X.columns}
        else:
            self.statistics_ = X.mode().to_dict("index")[0]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=True)

        if self.strategy == "constant":
            for col in X.columns:
                X[col] = X[col].cat.add_categories(self.statistics_[col])
        return X.fillna(self.statistics_)

    def get_dtypes(self, Xt, X, y=None):
        return Xt.dtypes.to_dict()


class SimpleNumericImputer(Imputer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "mean",
        "fill_value": 0,
        "verbose": 0,
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
        "verbose": 0,
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": CategoricalDistribution(["most_frequent", "constant"])
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY