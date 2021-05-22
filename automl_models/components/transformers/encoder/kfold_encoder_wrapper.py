import numpy as np
import pandas as pd

from typing import Optional, Union

from category_encoders.utils import convert_input, convert_input_vector
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator, TransformerMixin, clone

from ..utils import categorical_column_to_float


# KFoldEncoderWrapper from https://github.com/nyanp/nyaggle
# MIT License

# Copyright (c) 2020 nyanp

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class KFoldEncoderWrapper(BaseEstimator, TransformerMixin):
    """KFold Wrapper for sklearn like interface
    This class wraps sklearn's TransformerMixIn (object that has fit/transform/fit_transform methods),
    and call it as K-fold manner.
    Args:
        base_transformer:
            Transformer object to be wrapped.
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.
            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        return_same_type:
            If True, `transform` and `fit_transform` return the same type as X.
            If False, these APIs always return a numpy array, similar to sklearn's API.
    """

    def __init__(
        self,
        base_transformer,
        cv=5,
        return_same_type: bool = True,
        is_classification=True,
        random_state=None,
    ):
        self.cv = cv
        self.base_transformer = base_transformer
        self.__name__ = f"KFold{base_transformer.__class__.__name__}"

        self.n_splits = 5
        self.return_same_type = return_same_type
        self.is_classification = is_classification
        self.random_state = random_state

    def _pre_train(self, y, cat_columns):
        self.cv_ = check_cv(self.cv, y, classifier=self.is_classification)
        if isinstance(self.cv, int):
            self.cv_.random_state = self.random_state
            self.cv_.shuffle = True
        self.n_splits = self.cv_.get_n_splits()
        self.transformers_ = [
            self._clone_and_set_params(self.base_transformer, cat_columns)
            for _ in range(self.n_splits + 1)
        ]

    def _clone_and_set_params(self, transformer, cat_columns):
        transformer = clone(transformer)
        try:
            transformer.set_params(random_state=self.random_state)
        except Exception:
            pass
        try:
            transformer.set_params(cols=cat_columns)
        except Exception:
            pass
        return transformer

    def _fit_train(
        self, X: pd.DataFrame, y: Optional[pd.Series], groups=None, **fit_params
    ) -> pd.DataFrame:
        if y is None:
            X_ = self.transformers_[-1].transform(X)
            return X_

        X_ = X.copy()

        for i, (train_index, test_index) in enumerate(self.cv_.split(X_, y, groups)):
            self.transformers_[i].fit(
                X.iloc[train_index], y.iloc[train_index], **fit_params
            )
            X_.iloc[test_index, :] = self.transformers_[i].transform(X.iloc[test_index])
        self.transformers_[-1].fit(X, y, **fit_params)

        return X_

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit models for each fold.
        Args:
            X:
                Data
            y:
                Target
        Returns:
            returns the transformer object.
        """
        self.fit_transform(X, y)
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform X
        Args:
            X: Data
        Returns:
            Transformed version of X. It will be pd.DataFrame If X is `pd.DataFrame` and return_same_type is True.
        """
        is_pandas = isinstance(X, pd.DataFrame)
        if is_pandas:
            X_index = X.index
            X = X.reset_index(drop=True).apply(categorical_column_to_float)
        X_ = self._fit_train(X, None)
        if is_pandas:
            X_.index = X_index
        return X_ if self.return_same_type and is_pandas else X_.values

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: pd.Series = None,
        groups=None,
        **fit_params,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit models for each fold, then transform X
        Args:
            X:
                Data
            y:
                Target
            fit_params:
                Additional parameters passed to models
        Returns:
            Transformed version of X. It will be pd.DataFrame If X is `pd.DataFrame` and return_same_type is True.
        """
        is_pandas = True
        cat_columns = list(X.select_dtypes("category").columns)
        X_index = X.index
        X_new = X.reset_index(drop=True).apply(categorical_column_to_float)
        y = y.reset_index(drop=True)
        assert len(X_new) == len(y)
        self._pre_train(y, cat_columns)

        X_ = self._fit_train(X_new, y, groups=groups, **fit_params)

        X_.index = X_index
        return X_ if self.return_same_type and is_pandas else X_.values
