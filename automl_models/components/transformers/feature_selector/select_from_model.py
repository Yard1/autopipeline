#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

import numpy as np

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.utils import safe_mask
from sklearn.base import is_classifier
from sklearn.feature_selection import SelectFromModel as _SelectFromModel
from sklearn.feature_selection._from_model import _calculate_threshold

from .utils import lightgbm_fs_config, get_shap, get_tree_num
from ..utils import categorical_column_to_int_categories
from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ..transformer import DataType

import warnings


class PandasSHAPSelectFromModel(PandasDataFrameTransformerMixin, _SelectFromModel):
    def __init__(
        self,
        estimator,
        *,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
        n_estimators=100,
        random_state=None,
        n_jobs=None
    ):
        if estimator == "LGBMRegressor":
            estimator = LGBMRegressor(**lightgbm_fs_config)
        elif estimator == "LGBMClassifier":
            estimator = LGBMClassifier(**lightgbm_fs_config)
        super().__init__(
            estimator=estimator,
            threshold=threshold,
            prefit=prefit,
            norm_order=norm_order,
            max_features=max_features,
            importance_getter=importance_getter,
        )
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def _get_tree_num(self, n_feat):
        return get_tree_num(self.estimator, n_feat)

    def fit(self, X, y=None, **fit_params):
        self._is_classification_ = is_classifier(self.estimator)
        try:
            self.estimator.set_params(random_state=self.random_state)
        except Exception:
            pass
        try:
            self.estimator.set_params(n_jobs=self.n_jobs)
        except Exception:
            pass
        # if (
        #     self._is_classification_
        #     and "LGBM" in str(self.estimator)
        #     or "lightgbm" in str(type(self.estimator))
        # ):
        #     self.estimator.set_params(colsample_bytree=np.sqrt(X.shape[1]) / X.shape[1])

        # set n_estimators
        if self.n_estimators == "auto":
            estimators = self._get_tree_num(X.shape[1])
        else:
            estimators = self.n_estimators
        self.estimator.set_params(n_estimators=estimators)

        X = X.apply(categorical_column_to_int_categories)
        if DataType.is_categorical(y.dtype):
            y = categorical_column_to_int_categories(y).astype(np.uint16)
        super().fit(X=X, y=y, **fit_params)
        self.shap_imp_ = self._get_shap_imp(X, self.estimator_)
        return self

    def transform(self, X):
        mask = self.get_support()
        if not mask.any():
            warnings.warn(
                "No features were selected: either the data is"
                " too noisy or the selection test too strict.",
                UserWarning,
            )
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return X.iloc[:, safe_mask(X, mask)]

    def _get_shap_imp(self, X, estimator):
        return get_shap(estimator, X)

    @property
    def threshold_(self):
        scores = self.shap_imp_
        return _calculate_threshold(self.estimator, scores, self.threshold)

    def _get_support_mask(self):
        # SelectFromModel can directly call on transform.
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, "estimator_"):
            estimator = self.estimator_
        else:
            raise ValueError(
                "Either fit the model before transform or set"
                ' "prefit=True" while passing the fitted'
                " estimator to the constructor."
            )
        scores = self.shap_imp_
        threshold = _calculate_threshold(estimator, scores, self.threshold)
        if self.max_features is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind="mergesort")[
                : self.max_features
            ]
            mask[candidate_indices] = True
        else:
            mask = np.ones_like(scores, dtype=bool)
        mask[scores < threshold] = False
        return mask
