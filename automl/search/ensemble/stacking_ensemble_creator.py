from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
import gc
from abc import ABC

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold

from ...components.estimators.linear_model import LogisticRegressionCV, ElasticNetCV
from ...components.estimators.estimator import Estimator
from ...components.estimators.ensemble import StackingClassifier, StackingRegressor
from .ensemble_creator import EnsembleCreator
from .ensemble_strategy import EnsembleStrategy
from ...problems.problem_type import ProblemType

import logging

logger = logging.getLogger(__name__)


class StackingEnsembleCreator(EnsembleCreator):
    _returns_stacked_predictions: bool = False
    _ensemble_name: str = "Stack"

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy,
        problem_type: ProblemType,
        *,
        final_estimator: Optional[Estimator] = None,
    ) -> None:
        super().__init__(ensemble_strategy, problem_type)
        self.final_estimator = final_estimator

    @property
    def ensemble_class(self) -> type:
        return StackingClassifier if self.problem_type.is_classification() else StackingRegressor

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.final_estimator_ = self.final_estimator or ElasticNetCV(
                random_state=random_state,
                cv=KFold(shuffle=True, random_state=random_state),
            )
        elif self.problem_type.is_classification():
            self.final_estimator_ = self.final_estimator or LogisticRegressionCV(
                scoring=metric,
                random_state=random_state,
                cv=StratifiedKFold(shuffle=True, random_state=random_state),
            )
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
    ) -> Tuple[BaseEstimator, pd.DataFrame, pd.DataFrame]:
        assert "fold_predictions" in kwargs
        assert "refit_estimators" in kwargs
        assert "cv" in kwargs
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

        estimators = self._get_estimators_for_ensemble(
            trials_for_ensembling, current_stacking_level
        )
        if not estimators:
            raise ValueError("No estimators selected for stacking!")

        logger.debug(
            f"creating stacking classifier {self._ensemble_name} with {len(estimators)}"
        )
        ensemble = self.ensemble_class(
            estimators=estimators,
            final_estimator=self.final_estimator_(),
            cv=kwargs["cv"],
            n_jobs=None,
        )()
        logger.debug("ensemble created")
        gc.collect()
        logger.debug("fitting ensemble")
        print("fitting ensemble")
        ensemble.n_jobs = -1  # TODO make dynamic
        ensemble.fit(
            X,
            y,
            predictions=[
                kwargs["fold_predictions"].get(k, None) for k in self.trial_ids_for_ensembling_
            ]
            if kwargs["fold_predictions"]
            else None,
            refit_estimators=kwargs["refit_estimators"],
        )
        X_stack = ensemble.stacked_predictions_
        if X_test is not None:
            X_test_stack = ensemble.transform(X_test)
        else:
            X_test_stack = None
        return ensemble, X_stack, X_test_stack