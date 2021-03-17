import numpy as np
import pandas as pd

from typing import Optional, Union

from category_encoders.cat_boost import CatBoostEncoder as _CatBoostEncoder
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator, TransformerMixin, clone

from .encoder import Encoder
from ..transformer import DataType
from ..utils import categorical_column_to_int_categories, categorical_column_to_float
from ...component import ComponentLevel, ComponentConfig
from ....problems import ProblemType
from ....search.stage import AutoMLStage


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

    def _pre_train(self, y):
        self.cv_ = check_cv(self.cv, y, classifier=self.is_classification)
        if isinstance(self.cv, int):
            self.cv_.random_state = self.random_state
            self.cv_.shuffle = True
        self.n_splits = self.cv_.get_n_splits()
        self.transformers_ = [
            clone(self.base_transformer) for _ in range(self.n_splits + 1)
        ]

    def _fit_train(
        self, X: pd.DataFrame, y: Optional[pd.Series], groups=None, **fit_params
    ) -> pd.DataFrame:
        if y is None:
            X_ = self.transformers_[-1].transform(X)
            return self._post_transform(X_)

        X_ = X.copy()

        for i, (train_index, test_index) in enumerate(self.cv_.split(X_, y, groups)):
            self.transformers_[i].fit(
                X.iloc[train_index], y.iloc[train_index], **fit_params
            )
            X_.iloc[test_index, :] = self.transformers_[i].transform(X.iloc[test_index])
        self.transformers_[-1].fit(X, y, **fit_params)

        return X_

    def _post_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return X

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

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
        self._post_fit(self.fit_transform(X, y), y)
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
            X = X.apply(categorical_column_to_float).reset_index(drop=True)
        X_ = self._fit_train(X, None)
        X_ = self._post_transform(X_)
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
        X_index = X.index
        X = X.apply(categorical_column_to_float).reset_index(drop=True)
        y = y.reset_index(drop=True)
        assert len(X) == len(y)
        self._pre_train(y)

        is_pandas = isinstance(X, pd.DataFrame)
        X = convert_input(X)
        y = convert_input_vector(y, X.index)

        if y.isnull().sum() > 0:
            # y == null is regarded as test data
            X_ = X.copy()
            X_.loc[~y.isnull(), :] = self._fit_train(
                X[~y.isnull()], y[~y.isnull()], **fit_params
            )
            X_.loc[y.isnull(), :] = self._fit_train(X[y.isnull()], None, **fit_params)
        else:
            X_ = self._fit_train(X, y, groups=groups, **fit_params)

        X_ = self._post_transform(self._post_fit(X_, y))
        X_.index = X_index
        return X_ if self.return_same_type and is_pandas else X_.values


class CatBoostEncoderClassification(Encoder):
    _component_class = KFoldEncoderWrapper
    _default_parameters = {
        "base_transformer": _CatBoostEncoder(
            **{
                "verbose": 0,
                "cols": None,
                "drop_invariant": False,
                "return_df": True,
                "handle_unknown": "value",
                "handle_missing": "value",
                "random_state": None,
                "sigma": None,
                "a": 1,
            }
        ),
        "cv": 5,
        "return_same_type": True,
        "is_classification": True,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.COMMON
    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )


class CatBoostEncoderRegression(Encoder):
    _component_class = KFoldEncoderWrapper
    _default_parameters = {
        "base_transformer": _CatBoostEncoder(
            **{
                "verbose": 0,
                "cols": None,
                "drop_invariant": False,
                "return_df": True,
                "handle_unknown": "value",
                "handle_missing": "value",
                "random_state": None,
                "sigma": None,
                "a": 1,
            }
        ),
        "cv": 5,
        "return_same_type": True,
        "is_classification": False,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.COMMON
    _problem_types = {
        ProblemType.REGRESSION,
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )