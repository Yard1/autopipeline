import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasUnknownCategoriesDropper(
    PandasDataFrameTransformerMixin, TransformerMixin, BaseEstimator
):
    def __init__(self) -> None:
        super().__init__()

    def _validate_input(self, X: pd.DataFrame, in_fit):
        return X

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=False)
        X = X.copy()
        for col in X.select_dtypes("category"):
            diff = set(X[col].cat.categories) - set(self.categories_[col])
            if diff:
                X[col] = X[col].cat.remove_categories(list(diff))
            X[col] = X[col].cat.reorder_categories(self.categories_[col])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        X = self._validate_input(X, in_fit=True)
        X = X.copy()
        self.categories_ = {}
        for col in X.select_dtypes("category"):
            X[col] = X[col].cat.remove_unused_categories()
            self.categories_[col] = X[col].cat.categories
        return X

    def get_dtypes(self, Xt: pd.DataFrame, X, y=None):
        return Xt.dtypes.to_dict()
