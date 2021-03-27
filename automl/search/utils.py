import ray
import numpy as np
import os
import json
from unittest.mock import patch
import contextlib
import collections

from sklearn.metrics._scorer import (
    _MultimetricScorer,
    partial,
    _cached_call,
    _BaseScorer,
)
from ..components.component import Component


class MultimetricScorerWithErrorScore(_MultimetricScorer):
    def __init__(self, error_score="raise", **scorers):
        self._error_score = error_score
        self._scorers = scorers

    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)

        for name, scorer in self._scorers.items():
            try:
                if isinstance(scorer, _BaseScorer):
                    score = scorer._score(cached_call, estimator, *args, **kwargs)
                else:
                    score = scorer(estimator, *args, **kwargs)
            except Exception as e:
                if self._error_score == "raise":
                    raise e
                else:
                    score = self._error_score
            scores[name] = score
        return scores


def call_component_if_needed(possible_component, **kwargs):
    if isinstance(possible_component, Component):
        return possible_component(**kwargs)
    else:
        return possible_component


# TODO make this better
def optimized_precision(accuracy, recall, specificity):
    """
    Ranawana, Romesh & Palade, Vasile. (2006). Optimized Precision - A New Measure for Classifier Performance Evaluation. 2254 - 2261. 10.1109/CEC.2006.1688586.
    """
    try:
        return accuracy - (np.abs(specificity - recall) / (specificity + recall))
    except Exception:
        return accuracy


def flatten_iterable(x: list) -> list:
    if isinstance(x, list):
        return [a for i in x for a in flatten_iterable(i)]
    else:
        return [x]


def get_obj_name(obj):
    try:
        return obj.__name__
    except AttributeError:
        return obj.__class__.__name__


class ray_context:
    DEFAULT_CONFIG = {
        "ignore_reinit_error": True,
        "configure_logging": False,
        "include_dashboard": True,
        "_system_config": {
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}},
            )
        }
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
