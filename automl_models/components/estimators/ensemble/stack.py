from typing import Optional
import numpy as np
import pandas as pd
from functools import partial

from joblib import Parallel

from sklearn.ensemble import (
    StackingClassifier as _StackingClassifier,
    StackingRegressor as _StackingRegressor,
)
from sklearn.preprocessing import LabelEncoder  # TODO: consider PandasLabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.fixes import delayed
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier, clone
from sklearn.utils.validation import check_is_fitted, check_memory


from .utils import (
    call_method,
    ray_call_method,
    should_use_ray,
    put_args_if_ray,
    fit_single_estimator,
    ray_fit_single_estimator,
    fit_estimators,
    get_ray_pg,
    ray_pg_context,
    get_cv_predictions,
)
from ...utils import clone_with_n_jobs, ray_put_if_needed
from ...preprocessing import PrepareDataFrame
from ...flow import BasePipeline

import logging
import ray

logger = logging.getLogger(__name__)


def _fit_final_estimator(
    parallel, final_estimator, X_meta, y, sample_weight, fit_kwargs, pg=None
):
    if should_use_ray(parallel):
        estimator_ref = ray_put_if_needed(final_estimator)
        X_meta_ref = ray_put_if_needed(X_meta)
        y_ref = ray_put_if_needed(y)
        sample_weight_ref = ray_put_if_needed(sample_weight)
        fitted_estimator = [
            ray.get(
                ray_fit_single_estimator.options(
                    placement_group=pg, num_cpus=pg.bundle_specs[-1]["CPU"] if pg else 1
                ).remote(
                    estimator_ref, X_meta_ref, y_ref, sample_weight_ref, **fit_kwargs
                )
            )
        ]
    else:
        fitted_estimator = parallel(
            delayed(fit_single_estimator)(
                final_estimator,
                X_meta,
                y,
                sample_weight=sample_weight,
                **fit_kwargs,
            )
            for i in range(1)
        )
    return fitted_estimator


def _get_predictions(parallel, estimators, X, stack_method, pg=None):
    if should_use_ray(parallel):
        estimators = [
            (ray_put_if_needed(est), meth)
            for est, meth in zip(estimators, stack_method)
            if est != "drop"
        ]
        X_ref = ray_put_if_needed(X)
        predictions = ray.get(
            [
                ray_call_method.options(
                    placement_group=pg, num_cpus=pg.bundle_specs[-1]["CPU"] if pg else 1
                ).remote(est, meth, X_ref)
                for est, meth in estimators
            ]
        )
    else:
        predictions = parallel(
            delayed(call_method)(est, meth, X)
            for est, meth in zip(estimators, stack_method)
            if est != "drop"
        )
    return predictions


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

    def set_deep_final_estimator(
        self, estimator, passthrough: Optional[bool] = True, n_jobs: Optional[int] = 1
    ):
        if isinstance(self.final_estimator, DeepStackMixin):
            if passthrough is not None:
                self.passthrough = passthrough
            if n_jobs is not None:
                self.n_jobs = n_jobs
            return self.final_estimator.set_deep_final_estimator(estimator)
        self.final_estimator = estimator
        self.final_estimator_ = estimator
        return


# TODO handle Repeated CV (here and in trainable)
class PandasStackingClassifier(DeepStackMixin, _StackingClassifier):
    _is_ensemble = True

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
        min_n_jobs_per_estimator=1,
        max_n_jobs_per_estimator=-1,
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
        self.min_n_jobs_per_estimator = min_n_jobs_per_estimator
        self.max_n_jobs_per_estimator = max_n_jobs_per_estimator

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
        all_estimators = [clone(est) for est in all_estimators]
        self._validate_final_estimator()
        # else:
        #    self.final_estimator_ = self.final_estimator
        stack_method = [self.stack_method] * len(all_estimators)

        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(
            parallel,
            self.n_jobs,
            len(all_estimators),
            self.min_n_jobs_per_estimator,
            self.max_n_jobs_per_estimator,
        )
        X_ray, y_ray, sample_weight_ray = put_args_if_ray(parallel, X, y, sample_weight)
        with ray_pg_context(pg) as pg:

            # Fit the base estimators on the whole training data. Those
            # base estimators will be used in transform, predict, and
            # predict_proba. They are exposed publicly.
            # if not hasattr(self, "estimators_"):
            self.estimators_ = fit_estimators(
                parallel,
                all_estimators,
                X_ray,
                y_ray,
                sample_weight_ray,
                partial(
                    clone_with_n_jobs,
                    n_jobs=int(pg.bundle_specs[-1]["CPU"]) if pg else 1,
                ),
                pg=pg,
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
            memory = check_memory(self.memory)
            get_cv_predictions_cached = memory.cache(
                get_cv_predictions,
                ignore=["parallel", "verbose", "n_jobs", "X_ray", "y_ray", "pg"],
            )
            predictions = get_cv_predictions_cached(
                parallel=parallel,
                all_estimators=sorted(all_estimators, key=lambda x: str(x)),
                X=X,
                y=y,
                X_ray=X_ray,
                y_ray=y_ray,
                cv=cv,
                fit_params=fit_params,
                verbose=self.verbose,
                stack_method=self.stack_method_,
                n_jobs=self.n_jobs,
                pg=pg,
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
                fitted_estimator = _fit_final_estimator(
                    parallel,
                    self.final_estimator_,
                    X_meta,
                    y_ray,
                    sample_weight_ray,
                    fit_kwargs,
                    pg=pg,
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
        except Exception:
            pass
        if save_predictions:
            self.stacked_predictions_ = df
        if self.passthrough:
            df = pd.concat((X, df), axis=1)
        return df

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(
            parallel,
            self.n_jobs,
            len(self.estimators_),
            self.min_n_jobs_per_estimator,
            self.max_n_jobs_per_estimator,
        )
        with ray_pg_context(pg) as pg:
            predictions = _get_predictions(
                parallel, self.estimators_, X, self.stack_method_, pg=pg
            )
        return self._concatenate_predictions(X, predictions)


class PandasStackingRegressor(DeepStackMixin, _StackingRegressor):
    _is_ensemble = True

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
        min_n_jobs_per_estimator=1,
        max_n_jobs_per_estimator=-1,
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
        self.min_n_jobs_per_estimator = min_n_jobs_per_estimator
        self.max_n_jobs_per_estimator = max_n_jobs_per_estimator

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
        all_estimators = [clone(est) for est in all_estimators]
        self._validate_final_estimator()
        # else:
        #    self.final_estimator_ = self.final_estimator
        stack_method = [self.stack_method] * len(all_estimators)

        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(
            parallel,
            self.n_jobs,
            len(all_estimators),
            self.min_n_jobs_per_estimator,
            self.max_n_jobs_per_estimator,
        )
        X_ray, y_ray, sample_weight_ray = put_args_if_ray(parallel, X, y, sample_weight)
        with ray_pg_context(pg) as pg:
            # Fit the base estimators on the whole training data. Those
            # base estimators will be used in transform, predict, and
            # predict_proba. They are exposed publicly.
            # if not hasattr(self, "estimators_"):
            self.estimators_ = fit_estimators(
                parallel,
                all_estimators,
                X_ray,
                y_ray,
                sample_weight_ray,
                partial(
                    clone_with_n_jobs,
                    n_jobs=int(pg.bundle_specs[-1]["CPU"]) if pg else 1,
                ),
                pg=pg,
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
            memory = check_memory(self.memory)
            get_cv_predictions_cached = memory.cache(
                get_cv_predictions,
                ignore=["parallel", "verbose", "n_jobs", "X_ray", "y_ray", "pg"],
            )
            predictions = get_cv_predictions_cached(
                parallel=parallel,
                all_estimators=sorted(all_estimators, key=lambda x: str(x)),
                X=X,
                y=y,
                X_ray=X_ray,
                y_ray=y_ray,
                cv=cv,
                fit_params=fit_params,
                verbose=self.verbose,
                stack_method=self.stack_method_,
                n_jobs=self.n_jobs,
                pg=pg,
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
                fitted_estimator = _fit_final_estimator(
                    parallel,
                    self.final_estimator_,
                    X_meta,
                    y_ray,
                    sample_weight_ray,
                    fit_kwargs,
                    pg=pg,
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
        except Exception:
            pass
        if save_predictions:
            self.stacked_predictions_ = df
        if self.passthrough:
            df = pd.concat((X, df), axis=1)
        return df

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(
            parallel,
            self.n_jobs,
            len(self.estimators_),
            self.min_n_jobs_per_estimator,
            self.max_n_jobs_per_estimator,
        )
        with ray_pg_context(pg) as pg:
            predictions = _get_predictions(
                parallel, self.estimators_, X, self.stack_method_, pg=pg
            )
        return self._concatenate_predictions(X, predictions)
