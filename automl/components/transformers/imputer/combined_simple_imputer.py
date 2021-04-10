import numpy as np
from .imputer import Imputer
from ...component import ComponentLevel
from ...flow._column_transformer import PandasColumnTransformer, make_column_selector
from .simple_imputer import SimpleCategoricalImputer, SimpleNumericImputer

from ....utils.string import removeprefix
from ....search.distributions import CategoricalDistribution, FunctionDistribution

categorical_selector = make_column_selector(dtype_include="category")
numeric_selector = make_column_selector(dtype_exclude="category")


class PandasCombinedSimpleImputer(PandasColumnTransformer):
    def __init__(
        self,
        *,
        remainder="drop",
        numeric_strategy="mean",
        numeric_fill_value=0,
        categorical_strategy="most_frequent",
        categorical_fill_value="missing_value",
        verbose=0,
        copy=True,
        transformer_weights=None,
        n_jobs=None,
    ):
        self.transformers = [
            (
                "CategoricalImputer",
                SimpleCategoricalImputer(
                    strategy=categorical_strategy,
                    fill_value=categorical_fill_value,
                    copy=copy,
                    verbose=verbose,
                )(),
                categorical_selector,
            ),
            (
                "NumericImputer",
                SimpleNumericImputer(
                    strategy=numeric_strategy,
                    fill_value=numeric_fill_value,
                    copy=copy,
                    verbose=verbose,
                )(),
                numeric_selector,
            ),
        ]
        self.numeric_strategy = numeric_strategy
        self.numeric_fill_value = numeric_fill_value
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.copy = copy
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.remainder = remainder

    def set_params(self, **kwargs):
        categorical_imputer_kwargs = {
            f"CategoricalImputer__{removeprefix(k, 'categorical_')}": v
            for k, v in kwargs.items()
            if k in ("categorical_strategy, categorical_fill_value", "copy, verbose")
        }
        numeric_imputer_kwargs = {
            f"NumericImputer__{removeprefix(k, 'numeric_')}": v
            for k, v in kwargs.items()
            if k in ("numeric_strategy, numeric_fill_value", "copy, verbose")
        }
        kwargs = {**kwargs, **categorical_imputer_kwargs, **numeric_imputer_kwargs}
        return super().set_params(**kwargs)


def get_numeric_strategy(config, space):
    X = config.X
    if X is None:
        return CategoricalDistribution(["mean", "median"])
    missing_values = X.select_dtypes("number").isna().sum().sum()
    if missing_values == 0:
        return CategoricalDistribution(["mean"])
    return CategoricalDistribution(["mean", "median"])


def get_categorical_strategy(config, space):
    X = config.X
    if X is None:
        return CategoricalDistribution(["most_frequent", "constant"])
    missing_values = X.select_dtypes("category").isna().sum().sum()
    if missing_values == 0:
        return CategoricalDistribution(["most_frequent"])
    return CategoricalDistribution(["most_frequent", "constant"])


class CombinedSimpleImputer(Imputer):
    _component_class = PandasCombinedSimpleImputer
    _default_parameters = {
        "numeric_strategy": "mean",
        "numeric_fill_value": 0,
        "categorical_strategy": "most_frequent",
        "categorical_fill_value": "missing_value",
        "verbose": 0,
        "copy": True,
        "n_jobs": 1,
        "transformer_weights": None,
    }
    _default_tuning_grid = {
        "numeric_strategy": FunctionDistribution(get_numeric_strategy),
        "categorical_strategy": FunctionDistribution(get_categorical_strategy),
    }
    _component_level = ComponentLevel.NECESSARY
