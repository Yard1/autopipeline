import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasUnknownCategoriesDropper(PandasDataFrameTransformerMixin, TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def _validate_input(self, X, in_fit):
        return X

    def fit(self, X, y=None):
        X = self._validate_input(X, in_fit=True)
        self.categories_ = {}
        for col in X.select_dtypes("category"):
            self.categories_[col] = set(X[col].cat.categories)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=False)
        X_copied = False
        for col in X.select_dtypes("category"):
            diff = set(X[col].cat.categories) - set(self.categories_[col])
            if diff:
                if not X_copied:
                    X = X.copy()
                    X_copied = True
                X[col] = X[col].cat.remove_categories(list(diff))
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return X

    def get_dtypes(self, Xt, X, y=None):
        return Xt.dtypes.to_dict()
