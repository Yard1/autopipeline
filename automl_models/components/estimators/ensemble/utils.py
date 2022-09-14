import joblib
from joblib.parallel import Parallel
import numpy as np
import ray
import math
from ray.util.joblib import register_ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
    get_current_placement_group
)
from copy import deepcopy
from sklearn.base import clone, is_classifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._validation import (
    indexable,
    check_cv,
    _check_is_permutation,
    _num_samples,
    LabelEncoder,
    _fit_and_predict,
    sp,
)
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils.validation import check_is_fitted, NotFittedError, check_random_state
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.utils import _print_elapsed_time
from sklearn.utils.fixes import delayed
from pandas.api.types import is_integer_dtype, is_bool_dtype

from ...utils import (
    ray_get_if_needed,
    ray_put_if_needed,
    split_list_into_chunks,
    clone_with_n_jobs,
)
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


@ray.remote
def ray_fit_and_predict(*args, n_jobs=1, **kwargs):
    register_ray()
    with joblib.parallel_backend("ray", n_jobs=n_jobs):
        return _fit_and_predict(*args, **kwargs)


def _cross_val_predict_ray_remotes(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    verbose=0,
    fit_params=None,
    method="predict",
    placement_group=None,
    num_cpus=None,
    X_ref=None,
    y_ref=None,
):
    X_ref = X_ref or ray_put_if_needed(X)
    y_ref = y_ref or ray_put_if_needed(y)
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])
    # if not _check_is_permutation(test_indices, _num_samples(X)):
    #    raise ValueError("cross_val_predict only works for partitions")

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    y.index = list(X.index)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    ray_fit_and_predict_cpus = ray_fit_and_predict.options(
        placement_group=placement_group,
        num_cpus=num_cpus,
        placement_group_capture_child_tasks=True,
    )
    predictions = [
        ray_fit_and_predict_cpus.remote(
            clone(estimator) if not isinstance(estimator, ray.ObjectRef) else estimator,
            X,
            y,
            train,
            test,
            verbose,
            fit_params,
            method,
            n_jobs=num_cpus,
        )
        for train, test in splits
    ]
    return predictions, test_indices, encode


def _cross_val_predict_handle_predictions(y, predictions, test_indices, encode):
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]


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


def ray_cross_val_predict_repeated(
    estimators,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    verbose=0,
    fit_params=None,
    methods=["predict"],
    placement_group=None,
    num_cpus=None,
    X_ref=None,
    y_ref=None,
    return_type="concat",
):
    """sklearn cross_val_predict with support for repeated CV splitters"""
    assert return_type in ("concat", "lists", "refs")
    X_ref = X_ref or ray_put_if_needed(X)
    y_ref = y_ref or ray_put_if_needed(y)
    repeat_predictions = []
    if isinstance(cv, _RepeatedSplits):
        n_repeats = cv.n_repeats
    else:
        n_repeats = 1
    rng = check_random_state(cv.random_state)

    for estimator, method in zip(estimators, methods):
        for idx in range(n_repeats):
            if isinstance(cv, _RepeatedSplits):
                repeat_cv = cv.cv(random_state=rng, shuffle=True, **cv.cvargs)
            else:
                repeat_cv = cv
            repeat_predictions.append(
                _cross_val_predict_ray_remotes(
                    estimator,
                    X,
                    y,
                    groups=groups,
                    cv=repeat_cv,
                    verbose=verbose,
                    fit_params=fit_params,
                    method=method,
                    num_cpus=num_cpus,
                    placement_group=placement_group,
                    X_ref=X_ref,
                    y_ref=y_ref,
                )
            )

    predictions, test_indices, encodes = list(zip(*repeat_predictions))
    num_preds_per_estimator = len(predictions[0])
    predictions = [item for sublist in predictions for item in sublist]
    predictions, _ = ray.wait(predictions, num_returns=len(predictions))
    predictions = split_list_into_chunks(predictions, num_preds_per_estimator)
    if return_type == "concat":
        repeat_predictions = [
            _cross_val_predict_handle_predictions(
                y, ray.get(prediction), test_index, encode
            )
            for prediction, test_index, encode in zip(
                predictions, test_indices, encodes
            )
        ]
    elif return_type == "lists":
        repeat_predictions = [ray.get(prediction) for prediction in predictions]
    else:
        repeat_predictions = predictions

    if return_type == "concat" and n_repeats > 1:
        averaged_preds_list = []
        for chunk, estimator in zip(
            split_list_into_chunks(repeat_predictions, len(estimators)), estimators
        ):
            estimator = ray_get_if_needed(estimator)
            averaged_preds = np.mean(chunk, axis=0)
            if is_classifier(estimator) and (
                is_integer_dtype(chunk[0].dtype) or is_bool_dtype(chunk[0].dtype)
            ):
                averaged_preds = np.round(averaged_preds)
            averaged_preds.astype(chunk[0].dtype)
            averaged_preds_list.append(averaged_preds)
            repeat_predictions = averaged_preds_list
    return repeat_predictions


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
    parallel,
    all_estimators,
    X,
    y,
    sample_weight,
    clone_function,
    pg=None,
    X_ray=None,
    y_ray=None,
    sample_weight_ray=None,
):
    if pg or should_use_ray(parallel):
        cloned_estimators = [
            ray_put_if_needed(clone_function(est))
            for est in all_estimators
            if est != "drop"
        ]
        X_ref = ray_put_if_needed(X_ray or X)
        y_ref = ray_put_if_needed(y_ray or y)
        sample_weight_ref = ray_put_if_needed(sample_weight_ray or sample_weight)
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
    ) and not get_current_placement_group()


def put_args_if_ray(parallel: Parallel, *args):
    if should_use_ray(parallel):
        return (ray_put_if_needed(arg) for arg in args)
    return args


def get_ray_pg(parallel, n_jobs, n_estimators, min_n_jobs=1, max_n_jobs=-1):
    pg = None
    if should_use_ray(parallel):
        n_jobs = (
            min(1, n_jobs)
            if n_jobs and n_jobs >= 0
            else int(ray.cluster_resources()["CPU"])
        )
        max_cpus_per_node = min(node["Resources"].get("CPU", 1) for node in ray.nodes())
        assert min_n_jobs <= max_cpus_per_node
        n_jobs_per_estimator = max(1, min(n_jobs // n_estimators, max_cpus_per_node))
        n_jobs_per_estimator = int(pow(2, int(math.log(n_jobs_per_estimator, 2))))
        n_jobs_per_estimator = max(min_n_jobs, n_jobs_per_estimator)
        if max_n_jobs and max_n_jobs > 0:
            n_jobs_per_estimator = min(max_n_jobs, n_jobs_per_estimator)
        n_bundles = max(1, n_jobs // n_jobs_per_estimator)
        pg = placement_group([{"CPU": n_jobs_per_estimator}] * n_bundles)
        print(f"get_ray_pg: pg: {pg.bundle_specs} n_jobs: {n_jobs}")
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


def get_cv_predictions(
    parallel,
    all_estimators,
    X,
    y,
    cv,
    fit_params,
    verbose,
    stack_method,
    n_jobs,
    X_ray=None,
    y_ray=None,
    pg=None,
    groups=None,
    return_type="concat",
):
    print(f"getting cv predictions")
    if pg or should_use_ray(parallel):
        cloned_estimators = [
            (
                ray_put_if_needed(
                    clone_with_n_jobs(
                        est, n_jobs=int(pg.bundle_specs[-1]["CPU"]) if pg else 1
                    )
                ),
                meth,
            )
            for est, meth in zip(all_estimators, stack_method)
            if est != "drop"
        ]
        X_ref = ray_put_if_needed(X_ray or X)
        y_ref = ray_put_if_needed(y_ray or y)
        fit_params_ref = ray_put_if_needed(fit_params)
        estimators, methods = zip(*cloned_estimators)
        predictions = ray_cross_val_predict_repeated(
            estimators,
            X,
            y,
            cv=deepcopy(cv),
            methods=methods,
            fit_params=fit_params_ref,
            verbose=verbose,
            placement_group=pg,
            num_cpus=pg.bundle_specs[-1]["CPU"] if pg else 1,
            X_ref=X_ref,
            y_ref=y_ref,
            groups=groups,
            return_type=return_type,
        )
    else:
        predictions = parallel(
            delayed(cross_val_predict_repeated)(
                clone_with_n_jobs(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=n_jobs,
                fit_params=fit_params,
                verbose=verbose,
                groups=groups,
            )
            for est, meth in zip(all_estimators, stack_method)
            if est != "drop"
        )
    return predictions
