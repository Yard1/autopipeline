#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Daniel Homola <dani.homola@gmail.com>

Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/

License: BSD 3 clause
"""

import numpy as np
from boruta import BorutaPy
import warnings

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.utils import check_X_y
from sklearn.base import clone, is_classifier
from sklearn.utils.validation import check_random_state

from ...compatibility.pandas import PandasDataFrameTransformerMixin
from .utils import lightgbm_fs_config, get_shap, get_tree_num
from ..utils import categorical_column_to_int_categories
from ..transformer import DataType


class _BorutaSHAP(BorutaPy):
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
        if estimator == "LGBMRegressor":
            estimator = LGBMRegressor(**lightgbm_fs_config)
        elif estimator == "LGBMClassifier":
            estimator = LGBMClassifier(**lightgbm_fs_config)
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
        self._is_lightgbm = "LGBM" in str(self.estimator) or "lightgbm" in str(
            type(self.estimator)
        )

    def _get_shuffle(self, seq):
        self.random_state_.shuffle(seq)
        return seq

    def _fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)

        self.estimator_ = clone(self.estimator)
        self._is_classification_ = is_classifier(self.estimator_)

        # if self._is_classification_ and self._is_lightgbm:
        #     self.estimator_.set_params(
        #         colsample_bytree=np.sqrt(X.shape[1]) / X.shape[1]
        #     )

        X = X.apply(categorical_column_to_int_categories)
        if DataType.is_categorical(y.dtype):
            y = categorical_column_to_int_categories(y).astype(np.uint16)

        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X)
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        early_stopping = False
        if self.early_stopping:
            if self.n_iter_no_change >= self.max_iter:
                if self.verbose > 0:
                    print(
                        f"n_iter_no_change is bigger or equal to max_iter"
                        f"({self.n_iter_no_change} >= {self.max_iter}), "
                        f"early stopping will not be used."
                    )
            else:
                early_stopping = True

        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # early stopping vars
        _same_iters = 1
        _last_dec_reg = None
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != "auto":
            self.estimator_.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == "auto":
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator_.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            if self._is_lightgbm:
                self.estimator_.set_params(
                    random_state=self.random_state_.randint(0, 10000)
                )
            else:
                self.estimator_.set_params(random_state=self.random_state_)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

            # early stopping
            if early_stopping:
                if _last_dec_reg is not None and (_last_dec_reg == dec_reg).all():
                    _same_iters += 1
                    if self.verbose > 0:
                        print(
                            f"Early stopping: {_same_iters} out "
                            f"of {self.n_iter_no_change}"
                        )
                else:
                    _same_iters = 1
                    _last_dec_reg = dec_reg.copy()
                if _same_iters > self.n_iter_no_change:
                    break

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        self.importance_history_ = imp_history

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)

        if not np.sum(self.support_):
            if np.sum(self.support_ + self.support_weak_):
                self.support_ = self.support_ + self.support_weak_
            else:
                self.support_ = (self.support_ + 1).astype(bool)
        return self

    def transform(self, X):
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

        return self._transform(X, weak=False, return_df=True)

    def fit_transform(self, X, y):
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
        return self._transform(X, weak=False, return_df=True)

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y, dtype=None, force_all_finite=False)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError("The percentile should be between 0 and 100.")

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("Alpha should be between 0 and 1.")

    def _get_shap_imp(self, X, estimator):
        return get_shap(estimator, X)

    def _get_tree_num(self, n_feat):
        return get_tree_num(self.estimator_, n_feat)

    def _get_imp(self, X, y):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator_.fit(X, y)
        except Exception as e:
            raise ValueError(
                "Please check your X and y variable. The provided "
                "estimator cannot be fitted to your data.\n" + str(e)
            )
        try:
            imp = self._get_shap_imp(X, self.estimator_)
        except Exception:
            raise ValueError(
                "Only methods with feature_importance_ attribute "
                "are currently supported in BorutaPy."
            )
        return imp


class BorutaSHAP(PandasDataFrameTransformerMixin, _BorutaSHAP):
    def get_dtypes(self, Xt, X, y=None):
        return Xt.dtypes.to_dict()
