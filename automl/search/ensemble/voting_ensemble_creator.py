from copy import deepcopy
from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
import gc
from abc import ABC

from sklearn.base import BaseEstimator, clone

from ...components.estimators.ensemble import (
    VotingClassifier,
    VotingRegressor,
    GreedyVotingRegressor,
    GreedyVotingClassifier,
)
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
            self.weight_function_ = lambda trial: 1
            self.ensemble_args_ = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                self.weight_function_ = lambda trial: 1
            else:
                self.weight_function_ = lambda trial: 1
            self.ensemble_args_ = {"voting": "hard"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

    def fit_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: dict,
        pipeline_blueprint,
        metric_name: str,
        metric,
        random_state,
        current_stacking_level: int,
        previous_stack,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        X_test_original: Optional[pd.DataFrame],
        y_test_original: Optional[pd.Series],
        **kwargs,
    ) -> BaseEstimator:
        kwargs = self._treat_kwargs(kwargs)
        super().fit_ensemble(
            X,
            y,
            results,
            pipeline_blueprint,
            metric_name,
            metric,
            random_state,
            current_stacking_level,
            previous_stack,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
            **kwargs,
        )
        trials_for_ensembling = self.select_trials_for_ensemble(
            results, self.trial_ids_for_ensembling_
        )
        print(f"creating voting classifier {self._ensemble_name}")
        weights = [self.weight_function_(trial) for trial in trials_for_ensembling]
        trials_for_ensembling = [
            trial for idx, trial in enumerate(trials_for_ensembling) if weights[idx] is not None
        ]
        weights = [weight for weight in weights if weight is not None]
        print(f"getting estimators for {self._ensemble_name}")
        estimators = self._get_estimators_for_ensemble(
            trials_for_ensembling, current_stacking_level
        )
        if not estimators:
            raise ValueError("No estimators selected for ensembling!")
        print(f"final number of estimators: {len(estimators)}")
        ensemble = self.ensemble_class(
            estimators=estimators,
            weights=weights,
            n_jobs=-1,  # TODO make dynamic
            **{**(self.init_kwargs or {}), **self.ensemble_args_},
        )()
        if previous_stack:
            stacked_ensemble = clone(previous_stack)
            stacked_ensemble.set_deep_final_estimator(ensemble)
            ensemble = stacked_ensemble
        logger.debug("ensemble created")
        logger.debug("fitting ensemble")
        print(f"fitting ensemble {self}")
        ensemble.fit(
            X,
            y,
        )
        return ensemble


class VotingByMetricEnsembleCreator(VotingEnsembleCreator):
    _ensemble_name: str = "VotingByMetric"

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.weight_function_ = lambda trial: trial["metrics"][metric_name]
            self.ensemble_args_ = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                self.weight_function_ = lambda trial: trial["metrics"][metric_name]
            else:
                self.weight_function_ = lambda trial: trial["metrics"][metric_name]
            self.ensemble_args_ = {"voting": "hard"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")


class VotingSoftEnsembleCreator(VotingEnsembleCreator):
    _ensemble_name: str = "VotingSoft"

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.weight_function_ = lambda trial: 1
            self.ensemble_args_ = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                self.weight_function_ = (
                    lambda trial: 1
                    if hasattr(trial["estimator"], "predict_proba")
                    else 0
                )
            else:
                self.weight_function_ = (
                    lambda trial: 1
                    if hasattr(trial["estimator"], "predict_proba")
                    else 0
                )
            self.ensemble_args_ = {"voting": "soft"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")


class VotingSoftByMetricEnsembleCreator(VotingEnsembleCreator):
    _ensemble_name: str = "VotingSoftByMetric"

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.weight_function_ = lambda trial: trial["metrics"][metric_name]
            self.ensemble_args_ = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                self.weight_function_ = (
                    lambda trial: trial["metrics"][metric_name]
                    if hasattr(trial["estimator"], "predict_proba")
                    else 0
                )
            else:
                self.weight_function_ = (
                    lambda trial: trial["metrics"][metric_name]
                    if hasattr(trial["estimator"], "predict_proba")
                    else 0
                )
            self.ensemble_args_ = {"voting": "soft"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")


class GreedyEnsembleCreator(VotingSoftEnsembleCreator):
    _ensemble_name: str = "GreedyVoting"

    @property
    def ensemble_class(self) -> type:
        return (
            GreedyVotingClassifier
            if self.problem_type.is_classification()
            else GreedyVotingRegressor
        )

    def fit_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: dict,
        pipeline_blueprint,
        metric_name: str,
        metric,
        random_state,
        current_stacking_level: int,
        previous_stack,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        X_test_original: Optional[pd.DataFrame],
        y_test_original: Optional[pd.Series],
        **kwargs,
    ) -> BaseEstimator:
        kwargs = self._treat_kwargs(kwargs)
        assert "cv" in kwargs
        EnsembleCreator.fit_ensemble(
            self,
            X,
            y,
            results,
            pipeline_blueprint,
            metric_name,
            metric,
            random_state,
            current_stacking_level,
            previous_stack,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
            **kwargs,
        )
        trials_for_ensembling = self.select_trials_for_ensemble(
            results, self.trial_ids_for_ensembling_
        )
        print(f"creating voting classifier {self._ensemble_name}")
        trials_for_ensembling.sort(
            key=lambda trial: trial["metrics"][metric_name], reverse=True
        )
        print(f"getting estimators for {self._ensemble_name}")
        estimators = self._get_estimators_for_ensemble(
            trials_for_ensembling, current_stacking_level
        )
        if not estimators:
            raise ValueError("No estimators selected for ensembling!")
        print(f"final number of estimators: {len(estimators)}")

        ensemble = self.ensemble_class(
            estimators=estimators,
            random_state=random_state,
            cv=kwargs["cv"],
            scoring=metric,
            n_jobs=-1,  # TODO make dynamic
            ensemble_size=max(500, len(estimators)),
            n_iter_no_change=50,
            **{**(self.init_kwargs or {}), **self.ensemble_args_},
        )()
        if previous_stack:
            stacked_ensemble = clone(previous_stack)
            stacked_ensemble.set_deep_final_estimator(ensemble)
            ensemble = stacked_ensemble
        logger.debug("ensemble created")
        logger.debug("fitting ensemble")
        print(f"fitting ensemble {self}")
        ensemble.fit(
            X,
            y,
        )
        return ensemble
