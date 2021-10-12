import numpy as np
import pandas as pd
from copy import deepcopy

from joblib import Parallel

from sklearn.ensemble import (
    StackingClassifier as _StackingClassifier,
    StackingRegressor as _StackingRegressor,
)
from sklearn.preprocessing import LabelEncoder  # TODO: consider PandasLabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.fixes import delayed
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import _print_elapsed_time

from .utils import (
    get_cv_predictions,
    fit_single_estimator_if_not_fitted,
    call_method,
    cross_val_predict_repeated,
)
from ...utils import clone_with_n_jobs_1
from ...preprocessing import PrepareDataFrame
from ...flow import BasePipeline

import logging

logger = logging.getLogger(__name__)


def _fit_single_estimator(
    estimator,
    X,
    y,
    sample_weight=None,
    message_clsname=None,
    message=None,
    **fit_kwargs,
):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        estimator.__class__.__name__
                    )
                ) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y, **fit_kwargs)
    return estimator


class DeepStackMixin:
    def transform(self, X, deep: bool = False):
        ret = self._transform(X)
        if deep:
            if hasattr(self, "final_estimator_") and isinstance(
                self.final_estimator_, DeepStackMixin
            ):
                ret = self.final_estimator_.transform(ret)
        return ret

    def _get_deep_final_estimator(
        self, est, fitted: bool = False, up_to_stack: bool = False
    ):
        if isinstance(est, DeepStackMixin):
            return est.get_deep_final_estimator(fitted=fitted, up_to_stack=up_to_stack)
        if up_to_stack:
            return self
        return est

    def get_deep_final_estimator(self, fitted: bool = False, up_to_stack: bool = False):
        if fitted:
            check_is_fitted(self)
            return self._get_deep_final_estimator(
                self.final_estimator_, fitted=fitted, up_to_stack=up_to_stack
            )
        return self._get_deep_final_estimator(
            self.final_estimator, fitted=fitted, up_to_stack=up_to_stack
        )

    def set_deep_final_estimator(self, estimator):
        if isinstance(self.final_estimator, DeepStackMixin):
            return self.final_estimator.set_deep_final_estimator(estimator)
        self.final_estimator = estimator
        self.final_estimator_ = estimator
        return


# TODO handle Repeated CV (here and in trainable)
class PandasStackingClassifier(DeepStackMixin, _StackingClassifier):
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0,
        memory=None,
    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )
        self.memory = memory

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        save_predictions=False,
        fit_final_estimator=True,
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
        self.preprocessor_ = PrepareDataFrame(
            find_id_column=False,
            copy_X=False,
            missing_values_threshold=None,
            variance_threshold=None,
        )
        self.stacked_predictions_ = None
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()
        # else:
        #    self.final_estimator_ = self.final_estimator
        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        # if not hasattr(self, "estimators_"):
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone_with_n_jobs_1(est), X, y, sample_weight
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
        parallel = Parallel(n_jobs=self.n_jobs)
        predictions = parallel(
            delayed(cross_val_predict_repeated)(
                clone_with_n_jobs_1(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
            )
            for est, meth in zip(all_estimators, self.stack_method_)
            if est != "drop"
        )
        if save_predictions == "deep" and not isinstance(
            self.final_estimator_, DeepStackMixin
        ):
            save_predictions = True
        X_meta = self._concatenate_predictions(
            X,
            predictions,
            save_predictions if isinstance(save_predictions, bool) else False,
        )

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
            fit_kwargs = {}
            if save_predictions == "deep" and isinstance(
                self.final_estimator_, DeepStackMixin
            ):
                fit_kwargs = dict(save_predictions=save_predictions)
            if self.memory and not isinstance(
                self.final_estimator_, (BasePipeline, DeepStackMixin)
            ):
                self.final_estimator_ = BasePipeline(
                    [("final_estimator", self.final_estimator_)], memory=self.memory
                )
            fitted_estimator = parallel(
                delayed(_fit_single_estimator)(
                    self.final_estimator_,
                    X_meta,
                    y,
                    sample_weight=sample_weight,
                    **fit_kwargs,
                )
                for i in range(1)
            )
            self.final_estimator_ = fitted_estimator[0]

        return self

    def _concatenate_predictions(self, X, predictions, save_predictions: bool = False):
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
        index = X.index if hasattr(X, "index") else None
        X_meta = [
            pd.DataFrame(x, columns=X_meta_names[i], index=index)  # TODO: Better name
            for i, x in enumerate(X_meta)
        ]
        meta_df = pd.concat(X_meta, axis=1)
        try:
            df = self.preprocessor_.transform(meta_df)
        except Exception:
            df = self.preprocessor_.fit_transform(meta_df)
        try:
            df.index = X.index
        except:
            pass
        if save_predictions:
            self.stacked_predictions_ = df
        if self.passthrough:
            df = pd.concat((X, df), axis=1)
        return df


class PandasStackingRegressor(DeepStackMixin, _StackingRegressor):
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        n_jobs=None,
        passthrough=False,
        verbose=0,
        memory=None,
    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
        )
        self.memory = memory

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        save_predictions=False,
        fit_final_estimator=True,
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
        self.preprocessor_ = PrepareDataFrame(
            find_id_column=False,
            copy_X=False,
            missing_values_threshold=None,
            variance_threshold=None,
        )
        self.stacked_predictions_ = None
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()
        # else:
        #    self.final_estimator_ = self.final_estimator
        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        # if not hasattr(self, "estimators_"):
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone_with_n_jobs_1(est), X, y, sample_weight
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
        parallel = Parallel(n_jobs=self.n_jobs)
        predictions = parallel(
            delayed(cross_val_predict_repeated)(
                clone_with_n_jobs_1(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
            )
            for est, meth in zip(all_estimators, self.stack_method_)
            if est != "drop"
        )
        X_meta = self._concatenate_predictions(
            X, predictions, save_predictions=save_predictions
        )

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
            fit_kwargs = {}
            if save_predictions == "deep" and isinstance(
                self.final_estimator_, DeepStackMixin
            ):
                fit_kwargs = dict(save_predictions=save_predictions)
            if self.memory and not isinstance(
                self.final_estimator_, (BasePipeline, DeepStackMixin)
            ):
                self.final_estimator_ = BasePipeline(
                    [("final_estimator", self.final_estimator_)], memory=self.memory
                )
            fitted_estimator = parallel(
                delayed(_fit_single_estimator)(
                    self.final_estimator_,
                    X_meta,
                    y,
                    sample_weight=sample_weight,
                    **fit_kwargs,
                )
                for i in range(1)
            )
            self.final_estimator_ = fitted_estimator[0]

        return self

    def _concatenate_predictions(self, X, predictions, save_predictions: bool = False):
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
        index = X.index if hasattr(X, "index") else None
        X_meta = [
            pd.DataFrame(x, columns=X_meta_names[i], index=index)  # TODO: Better name
            for i, x in enumerate(X_meta)
        ]
        meta_df = pd.concat(X_meta, axis=1)
        try:
            df = self.preprocessor_.transform(meta_df)
        except Exception:
            df = self.preprocessor_.fit_transform(meta_df)
        try:
            df.index = X.index
        except:
            pass
        if save_predictions:
            self.stacked_predictions_ = df
        if self.passthrough:
            df = pd.concat((X, df), axis=1)
        return df
