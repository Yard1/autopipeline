from typing import Any, Dict, Tuple, Union
import ray
import numpy as np
import os
import json
from unittest.mock import patch
import contextlib
import time

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._validation import _score, _check_multimetric_scoring
from sklearn.utils.validation import check_is_fitted, NotFittedError

from sklearn.metrics._scorer import (
    _MultimetricScorer,
    partial,
    _cached_call,
    _BaseScorer,
    _PredictScorer,
    _ProbaScorer,
    _ThresholdScorer,
)


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


class ErrorScoreMixin:
    def _score(self, *args, **kwargs):
        try:
            super()._score(*args, **kwargs)
        except Exception:
            return self._kwargs["error_score"]


class PredictScorerWithErrorScore(ErrorScoreMixin, _PredictScorer):
    pass


class ProbaScorerWithErrorScore(ErrorScoreMixin, _ProbaScorer):
    pass


class ThresholdScorerWithErrorScore(ErrorScoreMixin, _ThresholdScorer):
    pass


def make_scorer_with_error_score(
    score_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    error_score=np.nan,
    **kwargs
):
    sign = 1 if greater_is_better else -1
    kwargs["error_score"] = error_score
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True," " but not both."
        )
    if needs_proba:
        cls = ProbaScorerWithErrorScore
    elif needs_threshold:
        cls = ThresholdScorerWithErrorScore
    else:
        cls = PredictScorerWithErrorScore
    return cls(score_func, sign, kwargs)