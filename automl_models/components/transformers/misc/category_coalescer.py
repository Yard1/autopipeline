from collections import defaultdict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from pandas.api.types import is_categorical_dtype
from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasCategoryCoalescer(
    PandasDataFrameTransformerMixin, TransformerMixin, BaseEstimator
):
    def __init__(self, minimum_fraction: float = 0.1) -> None:
        self.minimum_fraction = minimum_fraction

    def _validate_input(self, X: pd.DataFrame, in_fit):
        return X

    def _coalesce_categories(self, col: pd.Series):
        if not is_categorical_dtype(col.dtype):
            return col
        value_counts = col.value_counts(normalize=True)
        value_counts_below_fraction = value_counts[
            (value_counts < self.minimum_fraction) & (value_counts > 0)
        ]
        if value_counts_below_fraction.empty:
            return col
        categories_to_replace = list(value_counts_below_fraction.index)
        combined_category_name = "_".join(categories_to_replace)
        self.coalesced_categories_[col.name] = (
            categories_to_replace,
            combined_category_name,
        )
        if len(categories_to_replace) > 1:
            try:
                col = col.cat.add_categories([combined_category_name])
            except ValueError:
                pass
            col = col.replace({k: combined_category_name for k in categories_to_replace})
        col = col.cat.remove_unused_categories()
        return col

    def _coalesce_categories_transform(self, col: pd.Series):
        if col.name not in self.coalesced_categories_:
            return col
        assert is_categorical_dtype(col.dtype)
        categories_to_replace, combined_category_name = self.coalesced_categories_[
            col.name
        ]
        if len(categories_to_replace) > 1:
            try:
                col = col.cat.add_categories([combined_category_name])
            except ValueError:
                pass
            col = col.replace({k: combined_category_name for k in categories_to_replace})
        col = col.cat.remove_unused_categories()
        return col

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=True)
        X = X.copy()
        return X.apply(self._coalesce_categories_transform)

    def fit_transform(self, X: pd.DataFrame, y=None):
        assert 1 >= self.minimum_fraction >= 0
        X = self._validate_input(X, in_fit=True)
        X = X.copy()
        self.coalesced_categories_ = defaultdict(list)
        return X.apply(self._coalesce_categories)

    def get_dtypes(self, Xt: pd.DataFrame, X, y=None):
        return Xt.dtypes.to_dict()
