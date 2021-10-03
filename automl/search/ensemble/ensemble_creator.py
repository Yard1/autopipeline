import contextlib
from copy import deepcopy
import io
from typing import List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from abc import ABC
import gc

from sklearn.base import BaseEstimator

from ...components.estimators.ensemble.ensemble import Ensemble
from .ensemble_strategy import EnsembleStrategy
from ...problems.problem_type import ProblemType
from ..utils import stack_estimator

import logging

logger = logging.getLogger(__name__)


class EnsembleCreator(ABC):
    _returns_stacked_predictions: bool = False
    _ensemble_name: str = "None"

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy,
        problem_type: ProblemType,
    ) -> None:
        self.ensemble_strategy = ensemble_strategy
        self.problem_type = problem_type

    @property
    def ensemble_class(self) -> type:
        return None

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        return

    def _treat_kwargs(self, kwargs: dict) -> dict:
        kwargs = kwargs.copy()
        for k in kwargs:
            if isinstance(kwargs[k], dict):
                kwargs[k] = self._treat_kwargs(kwargs[k])
        return kwargs

    def _get_estimators_for_ensemble(
        self, trials_for_ensembling, current_stacking_level, previous_stack
    ):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            ret = [
                (
                    f"meta-{current_stacking_level}_{trial_result['trial_id']}",
                    # deepcopy(trial_result["estimator"]),
                    stack_estimator(trial_result["estimator"], previous_stack),
                )
                for trial_result in trials_for_ensembling
            ]
        return ret

    def select_trial_ids_for_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: dict,
        results_df: pd.DataFrame,
        pipeline_blueprint,
    ) -> List[BaseEstimator]:
        self.trial_ids_for_ensembling_ = self.ensemble_strategy.select_trial_ids(
            X=X,
            y=y,
            results=results,
            results_df=results_df,
            pipeline_blueprint=pipeline_blueprint,
        )
        return self.trial_ids_for_ensembling_

    def fit_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: dict,
        results_df: pd.DataFrame,
        pipeline_blueprint,
        metric_name: str,
        metric,
        random_state,
        current_stacking_level: int,
        previous_stack,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        X_test_original: Optional[pd.DataFrame] = None,
        y_test_original: Optional[pd.Series] = None,
        **kwargs,
    ) -> BaseEstimator:
        self._configure_ensemble(metric_name, metric, random_state)
        self.select_trial_ids_for_ensemble(
            X, y, results, results_df, pipeline_blueprint
        )
