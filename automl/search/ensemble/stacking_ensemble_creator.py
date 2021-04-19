from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
import gc
from abc import ABC

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold

from ...components.estimators.linear_model import (
    LogisticRegressionCV,
    LogisticRegression,
    ElasticNetCV,
    ElasticNet,
)
from ...components.transformers.misc.drop_columns import PandasSelectColumns
from ...components.estimators.estimator import Estimator
from ...components.estimators.ensemble import StackingClassifier, StackingRegressor
from ...components.transformers.feature_selector import (
    SHAPSelectFromModelClassification,
    SHAPSelectFromModelRegression,
)
from ...components.flow import Pipeline
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
        return (
            StackingClassifier
            if self.problem_type.is_classification()
            else StackingRegressor
        )

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.final_estimator_ = self.final_estimator or ElasticNet(
                random_state=random_state,
                # cv=KFold(shuffle=True, random_state=random_state),
            )
        elif self.problem_type.is_classification():
            self.final_estimator_ = self.final_estimator or LogisticRegression(
                # scoring=metric,
                random_state=random_state,
                # cv=StratifiedKFold(shuffle=True, random_state=random_state),
            )
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

    def _fit_ensemble(
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
                kwargs["fold_predictions"].get(k, None)
                for k in self.trial_ids_for_ensembling_
            ]
            if kwargs["fold_predictions"]
            else None,
            refit_estimators=kwargs["refit_estimators"],
            save_predictions=True,
        )
        X_stack = ensemble.stacked_predictions_
        if X_test is not None:
            X_test_stack = ensemble.transform(X_test)
        else:
            X_test_stack = None
        return ensemble, X_stack, X_test_stack

    def clear_stacked_predictions(self, ensemble):
        ensemble.stacked_predictions_ = None
        del ensemble.stacked_predictions_

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
        ensemble, X_stack, X_test_stack = self._fit_ensemble(
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
        self.clear_stacked_predictions(ensemble)
        return ensemble

    def fit_ensemble_and_return_stacked_preds(
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
        ensemble, X_stack, X_test_stack = self._fit_ensemble(
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
        self.clear_stacked_predictions(ensemble)
        return ensemble, X_stack, X_test_stack


class SelectFromModelStackingEnsembleCreator(StackingEnsembleCreator):
    _ensemble_name: str = "StackFSSelectFromModel"

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy,
        problem_type: ProblemType,
        *,
        final_estimator: Optional[Estimator] = None,
        max_estimators: int = 10,
    ) -> None:
        super().__init__(
            ensemble_strategy, problem_type, final_estimator=final_estimator
        )
        self.max_estimators = max_estimators

    @property
    def ensemble_class(self) -> type:
        return (
            StackingClassifier
            if self.problem_type.is_classification()
            else StackingRegressor
        )

    def _configure_ensemble(self, metric_name: str, metric, random_state):
        if self.problem_type == ProblemType.REGRESSION:
            self.final_estimator_ = Pipeline(
                steps=[
                    (
                        "SelectEstimators",
                        SHAPSelectFromModelRegression(
                            max_features=self.max_estimators, threshold=-np.inf
                        ),
                    ),
                    (
                        "FinalEstimator",
                        self.final_estimator
                        or ElasticNet(
                            random_state=random_state,
                            # cv=KFold(shuffle=True, random_state=random_state),
                        ),
                    ),
                ]
            )
        elif self.problem_type.is_classification():
            self.final_estimator_ = Pipeline(
                steps=[
                    (
                        "SelectEstimators",
                        SHAPSelectFromModelRegression(
                            max_features=self.max_estimators, threshold=-np.inf
                        ),
                    ),
                    (
                        "FinalEstimator",
                        self.final_estimator
                        or LogisticRegression(
                            # scoring=metric,
                            random_state=random_state,
                            # cv=StratifiedKFold(shuffle=True, random_state=random_state),
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

    def _fit_ensemble(
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
        ensemble, X_stack, X_stack_test = super()._fit_ensemble(
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
        # after the first fit, we can discard estimators that were not selected by feature selection
        columns_selected = ensemble.stacked_predictions_.columns[
            ensemble.final_estimator_.steps[0][1].get_support()
        ]
        estimator_names_to_keep = {
            "_".join(column_name.split("_")[:-1]) for column_name in columns_selected
        }
        indices_to_keep = {
            i
            for i, estimator in enumerate(ensemble.estimators)
            if estimator[0] in estimator_names_to_keep
        }
        print(indices_to_keep)
        estimator_names_to_remove = {
            name
            for name, _ in ensemble.estimators
            if name not in estimator_names_to_keep
        }
        print(estimator_names_to_remove)
        ensemble.estimators = [
            estimator
            for estimator in ensemble.estimators
            if estimator[0] in estimator_names_to_keep
        ]
        ensemble.estimators_ = [
            estimator
            for i, estimator in enumerate(ensemble.estimators_)
            if i in indices_to_keep
        ]
        for name in estimator_names_to_remove:
            del ensemble.named_estimators_[name]
        ensemble.stack_method_ = [
            stack_method
            for i, stack_method in enumerate(ensemble.stack_method_)
            if i in indices_to_keep
        ]
        print(ensemble.final_estimator_)
        ensemble.final_estimator_ = ensemble.final_estimator_.steps[0] = (
            ensemble.final_estimator_.steps[0][0],
            PandasSelectColumns(columns_selected),
        )
        ensemble.final_estimator = clone(ensemble.final_estimator_)
        return ensemble, X_stack, X_stack_test
