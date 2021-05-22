from ...flow.column_transformer import PandasColumnTransformer, make_column_selector
from .simple_imputer import PandasSimpleCategoricalImputer, PandasSimpleImputer

from ...utils import removeprefix

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
                PandasSimpleCategoricalImputer(
                    strategy=categorical_strategy,
                    fill_value=categorical_fill_value,
                    copy=copy,
                    verbose=verbose,
                ),
                categorical_selector,
            ),
            (
                "NumericImputer",
                PandasSimpleImputer(
                    strategy=numeric_strategy,
                    fill_value=numeric_fill_value,
                    copy=copy,
                    verbose=verbose,
                ),
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
