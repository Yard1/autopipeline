import pandas as pd

from sklearn.preprocessing import OrdinalEncoder as _OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from ...compatibility.pandas import (
    PandasDataFrameTransformerMixin,
)
from ...utils import validate_type


# TODO look into .codes
class PandasOrdinalEncoder(PandasDataFrameTransformerMixin, _OrdinalEncoder):
    def fit(self, X, y=None, **fit_params):
        """
        Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
        """
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        try:
            self.columns_ = X.columns
        except Exception:
            self.columns_ = None
        validate_type(X, "X", pd.DataFrame)

        self.categories_ = {}
        for col in X.select_dtypes("category"):
            self.categories_[col] = {
                cls: index for index, cls in enumerate(X[col].cat.categories)
            }

        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def transform(self, X):
        """
        Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        check_is_fitted(self)
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, columns=self.columns_)
        else:
            X = X.copy()
        validate_type(X, "X", pd.DataFrame)
        if not sorted(self.columns_) == sorted(X.columns):
            raise ValueError(
                f"Column mismatch. Expected {self.columns_}, got {X.columns}"
            )

        for col in X.select_dtypes("category"):
            diff = set(X[col].cat.categories) - set(self.categories_[col])
            if diff and self.handle_unknown == "error":
                raise ValueError(
                    f"'{col}' contains previously unseen labels: {list(diff)}"
                )
            X[col] = X[col].cat.rename_categories(self.categories_[col])

        return X

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.
        """
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, columns=self.columns_)
        else:
            X = X.copy()
        validate_type(X, "X", pd.DataFrame)
        if not sorted(self.columns_) == sorted(X.columns):
            raise ValueError(
                f"Column mismatch. Expected {self.columns_}, got {X.columns}"
            )

        for col in X.select_dtypes("category"):
            diff = set(X[col].cat.categories) - set(self.categories_[col].values())
            if diff and self.handle_unknown == "error":
                raise ValueError(
                    f"'{col}' contains previously unseen labels: {list(diff)}"
                )
            X[col] = X[col].cat.rename_categories(
                {v: k for k, v in self.categories_[col].items()}
            )

        return X
