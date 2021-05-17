from typing import Any, Dict, Tuple, Union
import ray
import numpy as np
import os
from copy import deepcopy
from unittest.mock import patch
import contextlib
import time
import traceback

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._validation import _score, _check_multimetric_scoring
from sklearn.utils.validation import check_is_fitted, NotFittedError

from sklearn.metrics._scorer import (
    _BaseScorer,
)
from .metrics.scorers import MultimetricScorerWithErrorScore
from ..components.component import Component

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


def get_obj_name(obj: Any) -> str:
    try:
        return obj.__name__
    except AttributeError:
        return obj.__class__.__name__


class ray_context:
    DEFAULT_CONFIG = {
        "ignore_reinit_error": True,
        "configure_logging": False,
        "include_dashboard": True,
        # "local_mode": True,
        # "num_cpus": 8,
    }

    def __init__(self, global_checkpoint_s=10, **ray_config):
        self.global_checkpoint_s = global_checkpoint_s
        self.ray_config = {**self.DEFAULT_CONFIG, **ray_config}
        self.ray_init = False

    def __enter__(self):
        self.ray_init = ray.is_initialized()
        if not self.ray_init:
            # TODO separate patch dict
            with patch.dict(
                "os.environ",
                {"TUNE_GLOBAL_CHECKPOINT_S": str(self.global_checkpoint_s)},
            ) if "TUNE_GLOBAL_CHECKPOINT_S" not in os.environ else contextlib.nullcontext():
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
        stack.final_estimator = estimator
        stack.final_estimator_ = estimator
        stack.passthrough = True
        estimator = stack
    return estimator
