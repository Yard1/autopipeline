#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

import numpy as np
import shap
from boruta import BorutaPy

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.utils import check_X_y
from sklearn.base import is_classifier
from sklearn.preprocessing import StandardScaler

from .feature_selector import FeatureSelector
from ..utils import categorical_column_to_int
from ...component import ComponentLevel
from ....problems import ProblemType

import warnings


class BorutaSHAP(BorutaPy):
    def __init__(
        self,
        estimator,
        n_estimators=100,
        perc=100,
        alpha=0.05,
        two_step=True,
        max_iter=100,
        random_state=None,
        verbose=0,
        early_stopping=False,
        n_iter_no_change=20,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            two_step=two_step,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
        )
        self._is_classification = is_classifier(self.estimator)

    def _fit(self, X, y):
        X = X.copy()
        X = X.apply(categorical_column_to_int)
        if self._is_classification and self._is_lightgbm:
            self.estimator.set_params(colsample_bytree=np.sqrt(X.shape[1]) / X.shape[1])
        return super()._fit(X, y)

    def transform(self, X, weak=False):
        """
        Reduces the input X to the features selected by Boruta.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X, weak, return_df=True)

    def fit_transform(self, X, y, weak=False):
        """
        Fits Boruta, then reduces the input X to the selected features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.

        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self._fit(X, y)
        return self._transform(X, weak, return_df=True)

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y, dtype=None)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError("The percentile should be between 0 and 100.")

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("Alpha should be between 0 and 1.")

    # from https://github.com/Ekeany/Boruta-Shap
    def _get_shap_imp(self, X, estimator):
        explainer = shap.TreeExplainer(
            estimator, feature_perturbation="tree_path_dependent"
        )
        if self._is_classification:
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

    def _get_imp(self, X, y):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator.fit(X, y)
        except Exception as e:
            raise ValueError(
                "Please check your X and y variable. The provided "
                "estimator cannot be fitted to your data.\n" + str(e)
            )
        try:
            imp = self._get_shap_imp(X, self.estimator)
        except Exception:
            raise ValueError(
                "Only methods with feature_importance_ attribute "
                "are currently supported in BorutaPy."
            )
        return imp


_lightgbm_rf_config = {
    "n_jobs": 1,
    "boosting_type": "rf",
    "max_depth": 5,
    "num_leaves": 32,
    "subsample": 0.632,
    "subsample_freq": 1,
    "verbose": -1
}


class BorutaSHAPClassification(FeatureSelector):
    _component_class = BorutaSHAP
    _default_parameters = {
        "estimator": LGBMClassifier(**_lightgbm_rf_config),
        "n_estimators": "auto",
        "perc": 100,
        "alpha": 0.05,
        "two_step": True,
        "max_iter": 100,
        "random_state": None,
        "verbose": 0,
        "early_stopping": True,
        "n_iter_no_change": 10,
    }
    _component_level = ComponentLevel.UNCOMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class BorutaSHAPRegression(FeatureSelector):
    _component_class = BorutaSHAP
    _default_parameters = {
        "estimator": LGBMRegressor(**_lightgbm_rf_config),
        "n_estimators": "auto",
        "perc": 100,
        "alpha": 0.05,
        "two_step": True,
        "max_iter": 100,
        "random_state": None,
        "verbose": 0,
        "early_stopping": True,
        "n_iter_no_change": 10,
    }
    _component_level = ComponentLevel.UNCOMMON
    _problem_types = {ProblemType.REGRESSION}
