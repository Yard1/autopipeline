import numpy as np
import ray
from copy import deepcopy
from sklearn.base import clone, ClassifierMixin, BaseEstimator, is_classifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils.validation import check_is_fitted, NotFittedError, check_random_state
from sklearn.model_selection._split import _RepeatedSplits
from pandas.api.types import is_integer_dtype, is_bool_dtype
from ...utils import split_list_into_chunks

import logging

logger = logging.getLogger(__name__)


class DummyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, id, preds, pred_probas) -> None:
        self.id = id
        self.preds = preds
        self.pred_probas = pred_probas

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.preds

    def predict_proba(self, X):
        return self.pred_probas


def fit_single_estimator_if_not_fitted(
    estimator,
    X,
    y,
    sample_weight=None,
    message_clsname=None,
    message=None,
    cloning_function=clone,
    force_refit=False,
):
    if hasattr(estimator, "_ray_cached_object"):
        cache = estimator._ray_cached_object
        estimator = cache.object
    else:
        cache = None
    try:
        assert not force_refit
        check_is_fitted(estimator)
        return estimator
    except (NotFittedError, AssertionError):
        print(f"fitting estimator {estimator}", flush=True)
        ret = _fit_single_estimator(
            cloning_function(estimator),
            X,
            y,
            sample_weight=sample_weight,
            message_clsname=message_clsname,
            message=message,
        )
        if cache:
            ret._ray_cached_object = cache
            cache.object = ret
        return ret


def _get_average_preds_from_repeated_cv(predictions: list, estimator):
    if not isinstance(predictions, list):
        return predictions
    if len(predictions) > 1:
        averaged_preds = np.mean(predictions, axis=0)
        if is_classifier(estimator) and (
            is_integer_dtype(predictions[0].dtype)
            or is_bool_dtype(predictions[0].dtype)
        ):
            averaged_preds = np.round(averaged_preds)
        return averaged_preds.astype(predictions[0].dtype)
    return predictions[0]


def _cross_val_predict_repeated(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    method="predict",
):
    """sklearn cross_val_predict with support for repeated CV splitters"""
    if isinstance(cv, _RepeatedSplits):
        repeat_predictions = []
        n_repeats = cv.n_repeats
        rng = check_random_state(cv.random_state)

        for idx in range(n_repeats):
            repeat_cv = cv.cv(random_state=rng, shuffle=True, **cv.cvargs)
            repeat_predictions.append(
                cross_val_predict(
                    estimator,
                    X,
                    y,
                    groups=groups,
                    cv=repeat_cv,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    fit_params=fit_params,
                    pre_dispatch=pre_dispatch,
                    method=method,
                )
            )

        averaged_preds = np.mean(repeat_predictions, axis=0)
        if is_classifier(estimator) and (
            is_integer_dtype(repeat_predictions[0].dtype)
            or is_bool_dtype(repeat_predictions[0].dtype)
        ):
            averaged_preds = np.round(averaged_preds)
        return averaged_preds.astype(repeat_predictions[0].dtype)

    return cross_val_predict(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        method=method,
    )


def get_cv_predictions(
    X,
    y,
    estimators,
    stack_methods,
    cv,
    predictions=None,
    n_jobs=None,
    fit_params=None,
    verbose=0,
):
    if predictions and not len(predictions) == len(estimators):
        raise ValueError(
            f"Length of predictions ({len(predictions)}) must be the same as the length of estimators ({len(estimators)})."
        )
    predictions_new = []
    fit_params = fit_params or {}
    for i, est_meth in enumerate(zip(estimators, stack_methods)):
        est, meth = est_meth
        if est == "drop":
            continue
        if predictions:
            prediction = predictions[i]
        else:
            prediction = {}

        if prediction and meth in prediction:
            predictions_new.append(
                _get_average_preds_from_repeated_cv(prediction[meth], est)
            )
        else:
            print(f"doing cv for {est}.{meth}")
            if hasattr(est, "_ray_cached_object"):
                cache = est._ray_cached_object
            else:
                cache = None
            try:
                predictions_new = ray.get(
                    cache.store_actor.get.remote(cache.key, "fold_predictions")
                )
            except Exception:
                predictions_new.append(
                    _cross_val_predict_repeated(
                        clone(est),
                        X,
                        y,
                        cv=deepcopy(cv),
                        method=meth,
                        n_jobs=n_jobs,
                        fit_params=fit_params,
                        verbose=verbose,
                    )
                )
                if cache:
                    ray.get(
                        cache.store_actor.put.remote(
                            cache.key, "fold_predictions", predictions_new
                        )
                    )
    return predictions_new


def call_method(obj, method_name, *args, **kwargs):
    return getattr(obj, method_name)(*args, **kwargs)
