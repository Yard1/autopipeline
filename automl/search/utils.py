from typing import Any, Dict, Tuple, Union
import ray
import numpy as np
import os
from copy import deepcopy
from unittest.mock import patch
import contextlib
import time
import traceback

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection._validation import _score, _check_multimetric_scoring
from sklearn.utils.validation import check_is_fitted, NotFittedError
from sklearn.model_selection._validation import (
    indexable,
    check_cv,
    is_classifier,
    check_scoring,
    _check_multimetric_scoring,
    _insert_error_scores,
    _normalize_score_results,
    _aggregate_score_dicts,
    _fit_and_score,
)
from sklearn.metrics._scorer import (
    _BaseScorer,
)
from .metrics.scorers import MultimetricScorerWithErrorScore
from ..components.component import Component

ray_fit_and_score = ray.remote(_fit_and_score)


def ray_cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
    X_ref=None,
    y_ref=None,
):
    """Fast cross validation with Ray, adapted from sklearn.validation.cross_validate"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    train_test = list(cv.split(X, y, groups))

    X_ref = X_ref if X_ref is not None else ray.put(X)
    y_ref = y_ref if y_ref is not None else ray.put(y)

    results_futures = [
        ray_fit_and_score.remote(
            clone(estimator),
            X_ref,
            y_ref,
            scorers,
            train,
            test,
            verbose,
            None,
            fit_params,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
        )
        for train, test in train_test
    ]

    results = ray.get(results_futures)

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    # added in automl
    ret["cv_indices"] = train_test

    return ret


def score_test(
    estimator: BaseEstimator,
    X,
    y,
    X_test,
    y_test,
    scoring: Dict[str, Union[str, _BaseScorer]],
    refit: bool = True,
    error_score=np.nan,
) -> Tuple[Dict[str, float], BaseEstimator]:
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        refit = True
    try:
        print(f"estimator {estimator.__class__.__name__} n_jobs: {estimator.n_jobs}")
    except Exception:
        pass
    if refit:
        st = time.time()
        estimator = clone(estimator)
        estimator.fit(X, y)
        print(f"fitting on train took {time.time()-st}")
    scoring = MultimetricScorerWithErrorScore(
        error_score=error_score, **_check_multimetric_scoring(estimator, scoring)
    )
    st = time.time()
    scores = _score(
        estimator,
        X_test,
        y_test,
        scoring,
        error_score="raise",
    )
    print(f"scoring took {time.time()-st}")
    return scores, estimator


ray_score_test = ray.remote(num_returns=2)(score_test)


def call_component_if_needed(possible_component, **kwargs):
    if isinstance(possible_component, Component):
        return possible_component(**kwargs)
    else:
        return possible_component


def flatten_iterable(x: list) -> list:
    if isinstance(x, list):
        return [a for i in x for a in flatten_iterable(i)]
    else:
        return [x]


def get_obj_name(obj: Any, deep: bool = True) -> str:
    if deep and hasattr(obj, "get_deep_final_estimator"):
        r = obj.get_deep_final_estimator(up_to_stack=True)
        if getattr(r.final_estimator, "_is_ensemble", False):
            return get_obj_name(r.final_estimator, deep=False)
        return get_obj_name(r, deep=False)
    try:
        return obj.__name__
    except AttributeError:
        return obj.__class__.__name__


class ray_context:
    DEFAULT_CONFIG = {
        "ignore_reinit_error": True,
        "namespace": "automl",
        "_temp_dir": "/home/ubuntu/automl/ray"
        # "configure_logging": False,
        # "include_dashboard": True,
        # "local_mode": True,
        # "num_cpus": 8,
    }

    def __init__(self, global_checkpoint_s=10, **ray_config):
        self.global_checkpoint_s = global_checkpoint_s
        self.ray_config = {**self.DEFAULT_CONFIG, **ray_config}
        self.ray_init = False

    def init(self):
        return self.__enter__()

    def __enter__(self):
        self.ray_init = ray.is_initialized()
        if not self.ray_init:
            # TODO separate patch dict
            with patch.dict(
                "os.environ",
                {
                    "TUNE_GLOBAL_CHECKPOINT_S": str(self.global_checkpoint_s),
                    "TUNE_RESULT_DELIM": "/",
                    "TUNE_FORCE_TRIAL_CLEANUP_S": "10",
                },
            ):
                ray.init(
                    **self.ray_config
                    # log_to_driver=self.verbose == 2
                )

    def __exit__(self, type, value, traceback):
        if not self.ray_init and ray.is_initialized():
            ray.shutdown()


def stack_estimator(estimator, stack):
    if stack:
        stack = deepcopy(stack)
        stack.passthrough = True
        stack.set_deep_final_estimator(estimator)
        estimator = stack
    return estimator
