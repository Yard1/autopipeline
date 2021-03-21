from copy import deepcopy
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict


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

        if meth in prediction:
            predictions_new.append(prediction[meth])
        else:
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
