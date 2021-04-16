from copy import deepcopy
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils.validation import check_is_fitted, NotFittedError

import logging

logger = logging.getLogger(__name__)

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
    try:
        assert not force_refit
        check_is_fitted(estimator)
        return estimator
    except (NotFittedError, AssertionError):
        print(f"fitting estimator {estimator}", flush=True)
        return _fit_single_estimator(
            cloning_function(estimator),
            X,
            y,
            sample_weight=sample_weight,
            message_clsname=message_clsname,
            message=message,
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
        raise ValueError(f"Length of predictions ({len(predictions)}) must be the same as the length of estimators ({len(estimators)}).")
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
            predictions_new.append(prediction[meth])
        else:
            print(f"doing cv for {est}.{meth}")
            predictions_new.append(
                cross_val_predict(
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
    return predictions_new


def call_method(obj, method_name, *args, **kwargs):
    return getattr(obj, method_name)(*args, **kwargs)
