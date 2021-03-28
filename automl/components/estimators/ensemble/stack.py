import numpy as np
import pandas as pd

import ray.exceptions
from joblib import Parallel

from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.preprocessing import LabelEncoder  # TODO: consider PandasLabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.fixes import delayed
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier, clone
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils.validation import check_is_fitted, NotFittedError

from .utils import get_cv_predictions, fit_single_estimator_if_not_fitted, call_method
from ....utils.estimators import clone_with_n_jobs_1
from ...preprocessing import PrepareDataFrame

import logging

logger = logging.getLogger(__name__)

# TODO consider RFE for LogisticRegression
class PandasStackingClassifier(StackingClassifier):
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        fit_final_estimator=True,
        refit_estimators=True,
        predictions=None,
    ):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        logger.debug("fitting stack", flush=True)
        self.preprocesser_ = PrepareDataFrame(find_id_column=False, copy_X=False)
        self.stacked_predictions_ = None
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        names, all_estimators = self._validate_estimators()
        # if not hasattr(self, "estimators_"):  # TODO Fix to make it work outside lib
        self._validate_final_estimator()
        # else:
        #    self.final_estimator_ = self.final_estimator
        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        # if not hasattr(self, "estimators_"):
        try:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    est, X, y, sample_weight, force_refit=refit_estimators
                )
                for est in all_estimators
                if est != "drop"
            )
        except:  # TODO is there a better way to catch exceptions here?
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    est,
                    X,
                    y,
                    sample_weight,
                    cloning_function=clone_with_n_jobs_1,
                    force_refit=refit_estimators,
                )
                for est in all_estimators
                if est != "drop"
            )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != "drop":
                self.named_estimators_[name_est] = self.estimators_[est_fitted_idx]
                est_fitted_idx += 1
            else:
                self.named_estimators_[name_est] = "drop"

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, "random_state") and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )
        # if not any(self.estimators[0][0] in col for col in X.columns):

        predictions = get_cv_predictions(
            X,
            y,
            all_estimators,
            self.stack_method_,
            cv,
            predictions=predictions,
            n_jobs=self.n_jobs,
            fit_params=fit_params,
            verbose=self.verbose,
        )
        X_meta = self._concatenate_predictions(X, predictions)
        # else:
        #    X_meta = X
        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth
            for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != "drop"
        ]

        if fit_final_estimator:
            try:
                self.final_estimator_.set_params(n_jobs=self.n_jobs)
            except ValueError:
                pass
            _fit_single_estimator(
                self.final_estimator_, X_meta, y, sample_weight=sample_weight
            )

        return self

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(call_method)(
                est,
                meth,
                X,
            )
            for est, meth in zip(self.estimators_, self.stack_method_)
            if est != "drop"
        )
        predictions = list(predictions)

        return self._concatenate_predictions(X, predictions)

    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)
        """
        X_meta = []
        X_meta_names = []
        for est_idx, preds in enumerate(predictions):
            # case where the the estimator returned a 1D array
            estimator_name = self.estimators[est_idx][0]
            if preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
                X_meta_names.append([f"{estimator_name}_-1"])
            else:
                if (
                    self.stack_method_[est_idx] == "predict_proba"
                    and len(self.classes_) == 2
                ):
                    # Remove the first column when using probabilities in
                    # binary classification because both features are perfectly
                    # collinear.
                    X_meta.append(preds[:, 1:])
                    X_meta_names.append([f"{estimator_name}_1"])
                else:
                    X_meta.append(preds)
                    X_meta_names.append(
                        [f"{estimator_name}_{i}" for i in range(preds.shape[1])]
                    )
        X_meta = [
            pd.DataFrame(x, columns=X_meta_names[i], index=X.index)  # TODO: Better name
            for i, x in enumerate(X_meta)
        ]
        meta_df = pd.concat(X_meta, axis=1)
        try:
            self.stacked_predictions_ = self.preprocesser_.transform(meta_df)
        except NotFittedError:
            self.stacked_predictions_ = self.preprocesser_.fit_transform(meta_df)
        self.stacked_predictions_.index = X.index
        df = self.stacked_predictions_
        if self.passthrough:
            df = pd.concat((X, df), axis=1)
        return df


class PandasStackingRegressor(StackingRegressor):
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        fit_final_estimator=True,
        refit_estimators=True,
        predictions=None,
    ):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        logger.debug("fitting stack", flush=True)
        self.preprocesser_ = PrepareDataFrame(find_id_column=False, copy_X=False)
        self.stacked_predictions_ = None
        names, all_estimators = self._validate_estimators()
        # if not hasattr(self, "estimators_"):  # TODO Fix to make it work outside lib
        self._validate_final_estimator()
        # else:
        #    self.final_estimator_ = self.final_estimator
        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        # if not hasattr(self, "estimators_"):
        try:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    est, X, y, sample_weight, force_refit=refit_estimators
                )
                for est in all_estimators
                if est != "drop"
            )
        except:  # TODO is there a better way to catch exceptions here?
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    est,
                    X,
                    y,
                    sample_weight,
                    cloning_function=clone_with_n_jobs_1,
                    force_refit=refit_estimators,
                )
                for est in all_estimators
                if est != "drop"
            )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != "drop":
                self.named_estimators_[name_est] = self.estimators_[est_fitted_idx]
                est_fitted_idx += 1
            else:
                self.named_estimators_[name_est] = "drop"

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, "random_state") and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )
        # if not any(self.estimators[0][0] in col for col in X.columns):

        predictions = get_cv_predictions(
            X,
            y,
            all_estimators,
            self.stack_method_,
            cv,
            predictions=predictions,
            n_jobs=self.n_jobs,
            fit_params=fit_params,
            verbose=self.verbose,
        )
        X_meta = self._concatenate_predictions(X, predictions)
        # else:
        #    X_meta = X
        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth
            for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != "drop"
        ]

        if fit_final_estimator:
            try:
                self.final_estimator_.set_params(n_jobs=self.n_jobs)
            except ValueError:
                pass
            _fit_single_estimator(
                self.final_estimator_, X_meta, y, sample_weight=sample_weight
            )

        return self

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(call_method)(
                est,
                meth,
                X,
            )
            for est, meth in zip(self.estimators_, self.stack_method_)
            if est != "drop"
        )
        predictions = list(predictions)

        return self._concatenate_predictions(X, predictions)

    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)
        """
        X_meta = []
        X_meta_names = []
        for est_idx, preds in enumerate(predictions):
            # case where the the estimator returned a 1D array
            estimator_name = self.estimators[est_idx][0]
            if preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
                X_meta_names.append([f"{estimator_name}_-1"])
            else:
                if (
                    self.stack_method_[est_idx] == "predict_proba"
                    and len(self.classes_) == 2
                ):
                    # Remove the first column when using probabilities in
                    # binary classification because both features are perfectly
                    # collinear.
                    X_meta.append(preds[:, 1:])
                    X_meta_names.append([f"{estimator_name}_1"])
                else:
                    X_meta.append(preds)
                    X_meta_names.append(
                        [f"{estimator_name}_{i}" for i in range(preds.shape[1])]
                    )
        X_meta = [
            pd.DataFrame(x, columns=X_meta_names[i], index=X.index)  # TODO: Better name
            for i, x in enumerate(X_meta)
        ]
        meta_df = pd.concat(X_meta, axis=1)
        try:
            self.stacked_predictions_ = self.preprocesser_.transform(meta_df)
        except NotFittedError:
            self.stacked_predictions_ = self.preprocesser_.fit_transform(meta_df)
        self.stacked_predictions_.index = X.index
        df = self.stacked_predictions_
        if self.passthrough:
            df = pd.concat((X, df), axis=1)
        return df
