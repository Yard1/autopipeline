from typing import Optional, Union, Tuple

import re
import pandas as pd
import numpy as np
from copy import deepcopy
import gc

from sklearn.base import BaseEstimator, clone

from ...components.estimators.linear_model import (
    LogisticRegressionCV,
    LogisticRegression,
    ElasticNetCV,
    ElasticNet,
)
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
from ...utils.memory import dynamic_memory_factory

from automl_models.components.transformers.misc.select_columns import (
    PandasSelectColumns,
)


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
        previous_stack,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        X_test_original: Optional[pd.DataFrame],
        y_test_original: Optional[pd.Series],
        **kwargs,
    ) -> Tuple[BaseEstimator, pd.DataFrame, pd.DataFrame]:
        assert "cv" in kwargs
        kwargs = self._treat_kwargs(kwargs)
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
            previous_stack,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
            **kwargs,
        )
        trials_for_ensembling = [results[k] for k in self.trial_ids_for_ensembling_]
        print(f"getting estimators for {self._ensemble_name}")
        estimators = self._get_estimators_for_ensemble(
            trials_for_ensembling, current_stacking_level
        )
        if not estimators:
            raise ValueError("No estimators selected for stacking!")

        print(
            f"creating stacking classifier {self._ensemble_name} with {len(estimators)}"
        )
        ensemble = self.ensemble_class(
            estimators=estimators,
            final_estimator=self.final_estimator_(),
            cv=kwargs["cv"],
            n_jobs=None,
            memory=dynamic_memory_factory(kwargs.get("cache", None)),
        )()
        if previous_stack:
            stacked_ensemble = clone(previous_stack)
            stacked_ensemble.set_deep_final_estimator(ensemble)
            ensemble = stacked_ensemble
        logger.debug("ensemble created")
        logger.debug("fitting ensemble")
        print(f"fitting ensemble {self}")
        ensemble.n_jobs = -1  # TODO make dynamic
        ensemble.fit(
            X,
            y,
            save_predictions="deep",
        )
        X_stack = ensemble.get_deep_final_estimator(
            fitted=True, up_to_stack=True
        ).stacked_predictions_

        if X_test_original is not None:
            # TODO optimize this
            X_test_stack = ensemble.transform(X_test_original, deep=True)
        else:
            X_test_stack = None

        return ensemble, X_stack, X_test_stack

    def clear_stacked_predictions(self, ensemble):
        del ensemble.get_deep_final_estimator(
            fitted=True, up_to_stack=True
        ).stacked_predictions_

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
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        X_test_original: Optional[pd.DataFrame],
        y_test_original: Optional[pd.Series],
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
            previous_stack,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
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
        previous_stack,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        X_test_original: Optional[pd.DataFrame],
        y_test_original: Optional[pd.Series],
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
            previous_stack,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
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
                            max_features=self.max_estimators, threshold="0.25*mean"
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
                        SHAPSelectFromModelClassification(
                            max_features=self.max_estimators, threshold="0.25*mean"
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
        previous_stack,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        X_test_original: Optional[pd.DataFrame],
        y_test_original: Optional[pd.Series],
        **kwargs,
    ) -> Tuple[BaseEstimator, pd.DataFrame, pd.DataFrame]:
        assert "cv" in kwargs
        kwargs = self._treat_kwargs(kwargs)
        super(StackingEnsembleCreator, self).fit_ensemble(
            X,
            y,
            results,
            results_df,
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
        trials_for_ensembling = [results[k] for k in self.trial_ids_for_ensembling_]
        print(f"getting estimators for {self._ensemble_name}")
        estimators = self._get_estimators_for_ensemble(
            trials_for_ensembling, current_stacking_level
        )
        if not estimators:
            raise ValueError("No estimators selected for stacking!")

        print(
            f"creating stacking classifier {self._ensemble_name} with {len(estimators)}"
        )
        ensemble = self.ensemble_class(
            estimators=estimators,
            final_estimator=self.final_estimator_(),
            cv=kwargs["cv"],
            n_jobs=None,
            memory=dynamic_memory_factory(kwargs.get("cache", None)),
        )()
        if previous_stack:
            stacked_ensemble = clone(previous_stack)
            stacked_ensemble.set_deep_final_estimator(ensemble)
            original_ensemble = ensemble
            ensemble = stacked_ensemble
        else:
            original_ensemble = ensemble
        logger.debug("ensemble created")
        logger.debug("fitting ensemble")
        print(f"fitting ensemble {self}")
        ensemble.n_jobs = -1  # TODO make dynamic
        try:
            ensemble.fit(
                X,
                y,
                save_predictions="deep",
            )
        except ValueError as e:
            # this is hacky but means we don't have to overwrite sklearn
            # SelectFromModel
            print(str(e))
            if "max_features" in str(e):
                max_features_to_set = int(
                    re.search(r"should be \d+ and (\d+)", str(e)).group(1)
                )
                print(f"Setting max_features to {max_features_to_set}")
                original_ensemble.final_estimator.set_params(
                    SelectEstimators__max_features=max_features_to_set
                )
                ensemble.fit(
                    X,
                    y,
                    save_predictions="deep",
                )
            else:
                raise e

        original_ensemble = ensemble.get_deep_final_estimator(
            fitted=True, up_to_stack=True
        )
        X_stack = original_ensemble.stacked_predictions_
        print(X_test.columns)
        if X_test_original is not None:
            # TODO optimize this
            X_test_stack = ensemble.transform(X_test_original, deep=True)
        else:
            X_test_stack = None

        # after the first fit, we can discard estimators that were not selected by feature selection
        columns_selected = original_ensemble.stacked_predictions_.columns[
            original_ensemble.final_estimator_.steps[0][1].get_support()
        ]
        estimator_names_to_keep = {
            "_".join(column_name.split("_")[:-1]) for column_name in columns_selected
        }
        indices_to_keep = {
            i
            for i, estimator in enumerate(original_ensemble.estimators)
            if estimator[0] in estimator_names_to_keep
        }
        print(indices_to_keep)
        estimator_names_to_remove = {
            name
            for name, _ in original_ensemble.estimators
            if name not in estimator_names_to_keep
        }
        print(estimator_names_to_remove)
        original_ensemble.estimators = [
            estimator
            for estimator in original_ensemble.estimators
            if estimator[0] in estimator_names_to_keep
        ]
        original_ensemble.estimators_ = [
            estimator
            for i, estimator in enumerate(original_ensemble.estimators_)
            if i in indices_to_keep
        ]
        for name in estimator_names_to_remove:
            del original_ensemble.named_estimators_[name]
        original_ensemble.stack_method_ = [
            stack_method
            for i, stack_method in enumerate(original_ensemble.stack_method_)
            if i in indices_to_keep
        ]
        original_ensemble.final_estimator_.steps[0] = (
            original_ensemble.final_estimator_.steps[0][0],
            PandasSelectColumns(columns_selected),
        )
        original_ensemble.final_estimator.steps[0] = (
            original_ensemble.final_estimator.steps[0][0],
            PandasSelectColumns(columns_selected),
        )

        return ensemble, X_stack, X_test_stack
