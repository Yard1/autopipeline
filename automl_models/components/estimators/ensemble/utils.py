from joblib.parallel import Parallel
import numpy as np
import ray
import gc
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
)
from copy import deepcopy
from sklearn.base import clone, ClassifierMixin, BaseEstimator, is_classifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils.validation import check_is_fitted, NotFittedError, check_random_state
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.utils import _print_elapsed_time
from sklearn.utils.fixes import delayed
from pandas.api.types import is_integer_dtype, is_bool_dtype

from ...utils import ray_put_if_needed

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
    if hasattr(estimator, "_ray_cached_object"):
        cache = estimator._ray_cached_object
    else:
        cache = None
    try:
        assert not force_refit
        check_is_fitted(estimator)
        return estimator
    except (NotFittedError, AssertionError):
        try:
            assert cache
            cache.clear_cache()
            ret = cache.object
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


def cross_val_predict_repeated(
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


ray_cross_val_predict_repeated = ray.remote(cross_val_predict_repeated)


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
                    cross_val_predict_repeated(
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


def fit_single_estimator(
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


ray_fit_single_estimator = ray.remote(fit_single_estimator)


def fit_estimators(
    parallel, all_estimators, X, y, sample_weight, clone_function, pg=None
):
    if should_use_ray(parallel):
        cloned_estimators = [
            ray_put_if_needed(clone_function(est))
            for est in all_estimators
            if est != "drop"
        ]
        X_ref = ray_put_if_needed(X)
        y_ref = ray_put_if_needed(y)
        sample_weight_ref = ray_put_if_needed(sample_weight)
        estimators = ray.get(
            [
                ray_fit_single_estimator.options(
                    placement_group=pg,
                    num_cpus=pg.bundle_specs[-1]["CPU"] if pg else 1,
                ).remote(est, X_ref, y_ref, sample_weight_ref)
                for est in cloned_estimators
            ]
        )
    else:
        estimators = parallel(
            delayed(fit_single_estimator)(clone_function(est), X, y, sample_weight)
            for est in all_estimators
            if est != "drop"
        )
    return estimators


def call_method(obj, method_name, *args, **kwargs):
    return getattr(obj, method_name)(*args, **kwargs)


ray_call_method = ray.remote(call_method)


def should_use_ray(parallel: Parallel) -> bool:
    return parallel.n_jobs not in (1, None) and (
        "ray" in parallel._backend.__class__.__name__.lower()  # or ray.is_initialized()
    )


def put_args_if_ray(parallel: Parallel, *args):
    if should_use_ray(parallel):
        return (ray_put_if_needed(arg) for arg in args)
    return args


def get_ray_pg(parallel, n_jobs, n_estimators):
    pg = None
    if should_use_ray(parallel):
        n_jobs = min(1, n_jobs) if n_jobs and n_jobs >= 0 else int(ray.cluster_resources()["CPU"])
        max_cpus_per_node = min(node["Resources"].get("CPU", 1) for node in ray.nodes())
        n_jobs_per_estimator = max(1, min(n_jobs // n_estimators, max_cpus_per_node))
        n_bundles = max(1, n_jobs // n_jobs_per_estimator)
        pg = placement_group([{"CPU": n_jobs_per_estimator}] * n_bundles)
        print(f"ray_get_pg: pg: {pg.bundle_specs} n_jobs: {n_jobs}")
    return pg


class ray_pg_context:
    def __init__(self, pg: PlacementGroup):
        self.pg = pg

    def __enter__(self) -> PlacementGroup:
        if self.pg:
            ray.get(self.pg.ready())
        return self.pg

    def __exit__(self, type, value, traceback):
        if self.pg:
            remove_placement_group(self.pg)
