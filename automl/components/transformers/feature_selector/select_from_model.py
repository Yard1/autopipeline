#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

import numpy as np
from numpy.linalg import norm
import shap
import contextlib

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.utils import safe_mask
from sklearn.base import clone, is_classifier
from sklearn.feature_selection import SelectFromModel as _SelectFromModel
from sklearn.utils.validation import check_random_state
from sklearn.feature_selection._from_model import _calculate_threshold

from .utils import lightgbm_rf_config as _lightgbm_rf_config
from .feature_selector import FeatureSelector
from ..utils import categorical_column_to_int_categories
from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ...component import ComponentLevel, ComponentConfig
from ...estimators.tree.tree_estimator import TreeEstimator
from ....problems import ProblemType
from ....search.stage import AutoMLStage

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
        random_state=None
    ):
        if estimator == "LGBMRegressor":
            estimator = LGBMRegressor(**_lightgbm_rf_config)
        elif estimator == "LGBMClassifier":
            estimator = LGBMClassifier(**_lightgbm_rf_config)
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

    def _get_tree_num(self, n_feat):
        depth = None
        try:
            depth = self.estimator.get_params()["max_depth"]
        except KeyError:
            warnings.warn(
                "The estimator does not have a max_depth property, as a result "
                " the number of trees to use cannot be estimated automatically."
            )
        if depth is None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        multi = (n_feat) / (np.sqrt(n_feat) * depth)
        n_estimators = int(multi * f_repr)
        return n_estimators

    def fit(self, X, y=None, **fit_params):
        self._is_classification_ = is_classifier(self.estimator)
        try:
            self.estimator.set_params(random_state=self.random_state)
        except:
            pass
        if (
            self._is_classification_
            and "LGBM" in str(self.estimator)
            or "lightgbm" in str(type(self.estimator))
        ):
            self.estimator.set_params(colsample_bytree=np.sqrt(X.shape[1]) / X.shape[1])

        # set n_estimators
        if self.n_estimators == "auto":
            estimators = self._get_tree_num(X.shape[1])
        else:
            estimators = self.n_estimators
        self.estimator.set_params(n_estimators=estimators)

        X = X.apply(categorical_column_to_int_categories)
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
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            explainer = shap.TreeExplainer(
                estimator, feature_perturbation="tree_path_dependent"
            )
            if self._is_classification_:
                # for some reason shap returns values wraped in a list of length 1
                shap_values = np.array(explainer.shap_values(X))
                if isinstance(shap_values, list):

                    class_inds = range(len(shap_values))
                    shap_imp = np.zeros(shap_values[0].shape[1])
                    for i, ind in enumerate(class_inds):
                        shap_imp += np.abs(shap_values[ind]).mean(0)
                    shap_values /= len(shap_values)

                elif len(shap_values.shape) == 3:
                    shap_values = np.abs(shap_values).sum(axis=0)
                    shap_values = shap_values.mean(0)

                else:
                    shap_values = np.abs(shap_values).mean(0)

            else:
                shap_values = explainer.shap_values(X)
                shap_values = np.abs(shap_values).mean(0)
            return shap_values

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


class SHAPSelectFromModelClassification(FeatureSelector):
    _component_class = PandasSHAPSelectFromModel
    _default_parameters = {
        "estimator": "LGBMClassifier",
        "threshold": "mean",
        "prefit": False,
        "norm_order": 1,
        "max_features": None,
        "importance_getter": "auto",
        "n_estimators": "auto",
        "random_state": None,
    }
    _component_level = ComponentLevel.UNCOMMON  # TODO: RARE
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        # RARE turns on Boruta which does the same thing but better (at a run time cost)
        return (
            super_check
            and (config.level is None or (config.level < ComponentLevel.RARE))
            and (
                config.estimator is None
                or not isinstance(config.estimator, TreeEstimator)
            )
        )


class SHAPSelectFromModelRegression(FeatureSelector):
    _component_class = PandasSHAPSelectFromModel
    _default_parameters = {
        "estimator": "LGBMRegressor",
        "threshold": "mean",
        "prefit": False,
        "norm_order": 1,
        "max_features": None,
        "importance_getter": "auto",
        "n_estimators": "auto",
        "random_state": None,
    }
    _component_level = ComponentLevel.UNCOMMON  # TODO: RARE
    _problem_types = {ProblemType.REGRESSION}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        # RARE turns on Boruta which does the same thing but better (at a run time cost)
        return (
            super_check
            and (config.level is None or (config.level < ComponentLevel.RARE))
            and (
                config.estimator is None
                or not isinstance(config.estimator, TreeEstimator)
            )
        )
