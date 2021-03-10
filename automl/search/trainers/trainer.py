from automl.components.estimators import estimator
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import gc

from copy import deepcopy, copy
from collections import defaultdict
from enum import Enum, auto

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.base import clone

import joblib
from ray.util.joblib import register_ray

register_ray()


from ..ensemble import EnsembleStrategy, RoundRobin, RoundRobinEstimator
from ..utils import ray_context
from ..tuners.tuner import Tuner
from ..tuners.OptunaTPETuner import OptunaTPETuner
from ..tuners.blendsearch import BlendSearchTuner
from ..tuners.BOHBTuner import BOHBTuner
from ..tuners.HEBOTuner import HEBOTuner
from ..blueprints.pipeline import create_pipeline_blueprint
from ..stage import AutoMLStage
from ..cv import get_cv_for_problem_type
from ...components.component import ComponentLevel, ComponentConfig
from ...components.estimators.ensemble.stack import PandasStackingClassifier
from ...components.estimators.linear_model import LogisticRegression
from ...problems.problem_type import ProblemType
from ...utils import validate_type
from ...utils.memory import dynamic_memory_factory

import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        problem_type: ProblemType,
        cv: Optional[Union[BaseCrossValidator, int]] = None,
        categorical_columns: Optional[list] = None,
        numeric_columns: Optional[list] = None,
        level: ComponentLevel = ComponentLevel.COMMON,
        tuner: Tuner = BlendSearchTuner,
        best_percentile: int = 5,
        max_ensemble_size: int = 10,
        ensemble_strategy: Optional[EnsembleStrategy] = None,
        stacking_level: int = 0,
        early_stopping: bool = False,
        cache: Union[str, bool] = False,
        random_state=None,
        tune_kwargs: dict = None,
    ) -> None:
        self.problem_type = problem_type
        self.cv = cv
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.level = level
        self.tuner = tuner
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.cache = cache
        self.best_percentile = best_percentile
        self.max_ensemble_size = max_ensemble_size
        self.ensemble_strategy = ensemble_strategy or RoundRobinEstimator()
        self.stacking_level = stacking_level
        self.tune_kwargs = tune_kwargs or {}

        # self.secondary_tuner = HEBOTuner
        self.secondary_tuner = None

    @property
    def last_tuner_(self):
        return self.tuners_[-1]

    @property
    def first_tuner_(self):
        return self.tuners_[0]

    def _get_cv(self, problem_type: ProblemType, cv: Union[BaseCrossValidator, int]):
        validate_type(cv, "cv", (BaseCrossValidator, int, None))
        return get_cv_for_problem_type(problem_type, n_splits=cv)

    def _fit_one_layer(self, X, y, groups=None):
        self.current_stacking_level += 1
        print(f"current_stacking_level: {self.current_stacking_level}")
        print(X.columns)
        categorical_columns = X.select_dtypes(include="category")
        numeric_columns = X.select_dtypes(exclude="category")
        pipeline_blueprint = create_pipeline_blueprint(
            problem_type=self.problem_type,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            level=self.level,
            X=X,
            y=y,
        )
        self.pipeline_blueprints_.append(pipeline_blueprint)
        tuner = self.tuner(
            problem_type=self.problem_type,
            pipeline_blueprint=pipeline_blueprint,
            random_state=self.random_state,
            cv=self.cv_,
            early_stopping=self.early_stopping,
            cache=self.cache,
        )
        self.tuners_.append(tuner)
        gc.collect()

        print("starting tuning")
        self.last_tuner_.fit(X, y, groups=groups)

        results = self.last_tuner_.analysis_.results
        results_df = self.last_tuner_.analysis_.results_df
        self.all_results_.append(results)
        self.all_results_df_.append(results_df)

        if "dataset_fraction" in results_df.columns:
            results_df = results_df[
                results_df["dataset_fraction"] >= 1.0
            ]  # TODO make dynamic
        percentile = np.percentile(results_df["mean_test_score"], self.best_percentile)

        if self.secondary_tuner is not None:
            self._run_secondary_tuning(
                X, y, pipeline_blueprint, percentile, groups=groups
            )
        print("tuning complete")
        configurations_for_ensembling = self._select_configurations_for_ensembling(
            results, results_df, pipeline_blueprint, percentile
        )
        ensemble = self._create_ensemble(
            configurations_for_ensembling, pipeline_blueprint
        )
        print("ensemble created")
        gc.collect()
        print("fitting ensemble")
        ensemble.n_jobs = -1  # TODO make dynamic
        with joblib.parallel_backend("ray"):
            ensemble.fit(
                X,
                y,
                fit_final_estimator=self.current_stacking_level >= self.stacking_level,
            )
        ensemble.n_jobs = None
        self.ensembles_.append(ensemble)
        if self.current_stacking_level >= self.stacking_level:
            # rfecv = RFECV(ensemble, step=1, cv=self.cv_)
            print("fitting final ensemble", flush=True)
            self.final_ensemble_ = self._create_final_ensemble()
            return
        X_stack = ensemble.stacked_predictions_
        self.meta_columns_.extend(X_stack.columns)
        return self._fit_one_layer(pd.concat((X, X_stack), axis=1), y, groups=groups)

    def _create_ensemble(
        self, configurations_for_ensembling, pipeline_blueprint, final_estimator=None
    ):
        default_config = {
            k: "passthrough"
            for k, v in pipeline_blueprint.get_all_distributions(
                use_extended=True
            ).items()
        }
        estimators = [
            (
                f"meta-{self.current_stacking_level}_{idx}",
                self._create_estimator(
                    config,
                    default_config=default_config,
                    pipeline_blueprint=pipeline_blueprint,
                ),
            )
            for idx, config in enumerate(configurations_for_ensembling)
        ]
        final_estimator = final_estimator or LogisticRegression()()
        print("creating stacking classifier")
        return PandasStackingClassifier(
            estimators=estimators, final_estimator=final_estimator, n_jobs=None
        )

    def _create_final_ensemble(self):
        if len(self.ensembles_) <= 1:
            return self.ensembles_[0]
        cloned_ensembles = [deepcopy(ensemble) for ensemble in self.ensembles_]
        for idx in range(0, len(cloned_ensembles) - 1):
            cloned_ensembles[idx].passthrough = True
            cloned_ensembles[idx].final_estimator = cloned_ensembles[idx + 1]
            cloned_ensembles[idx].final_estimator_ = cloned_ensembles[idx + 1]
        return cloned_ensembles[0]

    def _run_secondary_tuning(self, X, y, pipeline_blueprint, percentile, groups=None):
        groupby_list = [
            f"config.{k}" for k in pipeline_blueprint.get_all_distributions().keys()
        ]
        groupby_list.reverse()
        grouped_results_df = self.all_results_df_[-1].groupby(by=groupby_list)
        for name, group in grouped_results_df:
            if group["mean_test_score"].max() >= percentile:
                group_trial_ids = set(group.index)
                known_points = [
                    (
                        {
                            config_key: config_value
                            for config_key, config_value in v["config"].items()
                            if config_value != "passthrough"
                        },
                        v["mean_test_score"],
                    )
                    for k, v in self.all_results_[-1].items()
                    if k in group_trial_ids
                ]
                print(known_points[0])
                self.secondary_tuners_[self.current_stacking_level].append(
                    self.secondary_tuner(
                        problem_type=self.problem_type,
                        pipeline_blueprint=pipeline_blueprint,
                        random_state=self.random_state,
                        cv=self.cv_,
                        early_stopping=False,
                        cache=self.cache,
                        known_points=known_points,
                    )
                )
                self.secondary_tuners_[self.current_stacking_level][-1].fit(
                    X, y, groups=groups
                )
                secondary_results_df = self.secondary_tuners_[
                    self.current_stacking_level
                ][-1].analysis_.results_df
                if "dataset_fraction" in secondary_results_df.columns:
                    secondary_results_df = secondary_results_df[
                        secondary_results_df["dataset_fraction"]
                        >= 1.0  # TODO make dynamic
                    ]
                self.all_results_[-1].update(
                    self.secondary_tuners_[self.current_stacking_level][
                        -1
                    ].analysis_.results
                )
                self.all_results_df_[-1] = self.all_results_df_[-1].append(
                    secondary_results_df
                )

    def _create_estimator(self, config, default_config, pipeline_blueprint):
        config = {**default_config, **config}
        config.pop("dataset_fraction", None)
        estimator = pipeline_blueprint(random_state=self.random_state)
        config_called = self.last_tuner_._treat_config(config)
        estimator.set_params(**config_called)
        estimator.memory = dynamic_memory_factory(True)  # TODO make dynamic
        return estimator

    def _select_configurations_for_ensembling(
        self, results: dict, results_df: pd.DataFrame, pipeline_blueprint, percentile
    ) -> list:
        return self.ensemble_strategy.select_configurations(
            results=results,
            results_df=results_df,
            configurations_to_select=self.max_ensemble_size,
            pipeline_blueprint=pipeline_blueprint,
            percentile=percentile,
        )

    def fit(self, X, y, groups=None):
        self.current_stacking_level = -1

        self.cv_ = self._get_cv(self.problem_type, self.cv)
        self.pipeline_blueprints_ = []
        self.tuners_ = []
        self.secondary_tuners_ = defaultdict(list)
        self.all_results_ = []
        self.all_results_df_ = []
        self.ensembles_ = []
        self.meta_columns_ = []

        with ray_context(
            global_checkpoint_s=self.tune_kwargs.pop("TUNE_GLOBAL_CHECKPOINT_S", 10)
        ):
            return self._fit_one_layer(X, y, groups=groups)

        return self
