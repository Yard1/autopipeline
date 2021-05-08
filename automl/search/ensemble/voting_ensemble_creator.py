from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
import gc
from abc import ABC

from sklearn.base import BaseEstimator

from ...components.estimators.ensemble import VotingClassifier, VotingRegressor
from .ensemble_creator import EnsembleCreator
from ...problems.problem_type import ProblemType

import logging

logger = logging.getLogger(__name__)


class VotingEnsembleCreator(EnsembleCreator):
    _returns_stacked_predictions: bool = False
    _ensemble_name: str = "VotingUniform"

    @property
    def ensemble_class(self) -> type:
        return (
            VotingClassifier
            if self.problem_type.is_classification()
            else VotingRegressor
        )

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.weight_function_ = (
                lambda trial: 1 if trial["metrics"][metric_name] > 0.5 else 0
            )
            self.ensemble_args_ = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                self.weight_function_ = (
                    lambda trial: 1 if trial["metrics"][metric_name] > 0.5 else 0
                )
            else:
                self.weight_function_ = (
                    lambda trial: 1 if trial["metrics"][metric_name] > 0 else 0
                )
            self.ensemble_args_ = {"voting": "hard"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

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
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        **kwargs,
    ) -> BaseEstimator:
        super().fit_ensemble(
            X,
            y,
            results,
            results_df,
            pipeline_blueprint,
            metric_name,
            metric,
            random_state,
            current_stacking_level,
            X_test=X_test,
            y_test=y_test,
            **kwargs,
        )
        trials_for_ensembling = [results[k] for k in self.trial_ids_for_ensembling_]
        logger.debug(f"creating voting classifier {self._ensemble_name}")
        weights = [self.weight_function_(trial) for trial in trials_for_ensembling]
        trials_for_ensembling = [
            results[k]
            for idx, k in enumerate(self.trial_ids_for_ensembling_)
            if weights[idx] > 0
        ]
        weights = [weight for weight in weights if weight > 0]
        estimators = self._get_estimators_for_ensemble(
            trials_for_ensembling, current_stacking_level
        )
        if not estimators:
            raise ValueError("No estimators selected for ensembling!")
        logger.debug(f"final number of estimators: {len(estimators)}")
        ensemble = self.ensemble_class(
            estimators=estimators,
            weights=weights,
            n_jobs=None,
            **self.ensemble_args_,
        )()
        logger.debug("ensemble created")
        gc.collect()
        logger.debug("fitting ensemble")
        print("fitting ensemble")
        ensemble.n_jobs = -1  # TODO make dynamic
        ensemble.fit(
            X,
            y,
            refit_estimators=False,
        )
        test_predictions = kwargs.get("test_predictions", None)
        if test_predictions:
            ensemble._saved_test_predictions = [
                test_predictions.get(trial["trial_id"], None)
                for trial in trials_for_ensembling
            ]
        return ensemble


class VotingByMetricEnsembleCreator(VotingEnsembleCreator):
    _ensemble_name: str = "VotingByMetric"

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.weight_function_ = (
                lambda trial: trial["metrics"][metric_name]
                if trial["metrics"][metric_name] > 0.5
                else 0
            )
            self.ensemble_args_ = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                self.weight_function_ = (
                    lambda trial: trial["metrics"][metric_name]
                    if trial["metrics"][metric_name] > 0.5
                    else 0
                )
            else:
                self.weight_function_ = (
                    lambda trial: trial["metrics"][metric_name]
                    if trial["metrics"][metric_name] > 0
                    else 0
                )
            self.ensemble_args_ = {"voting": "hard"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")
