import contextlib
from copy import deepcopy
import io
from typing import List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from abc import ABC

from sklearn.base import BaseEstimator

from ...components.estimators.ensemble.ensemble import Ensemble
from .ensemble_strategy import EnsembleStrategy
from ...problems.problem_type import ProblemType

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

    def _get_estimators_for_ensemble(
        self, trials_for_ensembling, current_stacking_level
    ):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return [
                (
                    f"meta-{current_stacking_level}_{trial_result['trial_id']}",
                    deepcopy(trial_result["estimator"]),
                )
                for trial_result in trials_for_ensembling
            ]

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
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        **kwargs,
    ) -> BaseEstimator:
        self._configure_ensemble(metric_name, metric, random_state)
        self.select_trial_ids_for_ensemble(X, y, results, results_df, pipeline_blueprint)
