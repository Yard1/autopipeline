from sklearn.utils.validation import check_is_fitted
from automl.utils.display import IPythonDisplay
from sklearn.base import clone
from typing import List, Optional, Union, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import gc

from copy import deepcopy
from collections import defaultdict

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.preprocessing import LabelEncoder
import scipy.optimize
import contextlib
import io

import joblib
import ray
from ray.util.joblib import register_ray

from ...utils.joblib_backend.ray_backend import (
    multiprocessing_cache,
    register_ray_caching,
)

register_ray()
register_ray_caching()

from ..ensemble import (
    EnsembleStrategy,
    RoundRobin,
    EnsembleBest,
    RoundRobinEstimator,
    EnsembleCreator,
    VotingByMetricEnsembleCreator,
    VotingEnsembleCreator,
    StackingEnsembleCreator,
    SelectFromModelStackingEnsembleCreator,
    VotingSoftEnsembleCreator,
    VotingSoftByMetricEnsembleCreator,
)
from ..ensemble.ray import (
    ray_fit_ensemble,
    ray_fit_ensemble_and_return_stacked_preds_remote,
)
from ..utils import score_test, stack_estimator
from ..metrics.metrics import optimized_precision
from ..tuners.tuner import Tuner
from ..tuners.OptunaTPETuner import OptunaTPETuner
from ..tuners.blendsearch import BlendSearchTuner
from ..tuners.utils import treat_config
from ..blueprints.pipeline import create_pipeline_blueprint
from ..cv import get_cv_for_problem_type
from ...components.component import ComponentLevel, ComponentConfig
from ...components.estimators.linear_model import LogisticRegressionCV, ElasticNetCV
from ...problems.problem_type import ProblemType
from ...search.tuners.trainable import SklearnTrainable
from ...utils import validate_type
from ...utils.memory import dynamic_memory_factory
from ...components.flow.pipeline.base_pipeline import TopPipeline

from sklearn.metrics import matthews_corrcoef, recall_score, make_scorer
from deslib.des import METADES

from automl_models.components.estimators.ensemble import (
    PandasStackingClassifier,
    PandasStackingRegressor,
    PandasVotingClassifier,
    PandasVotingRegressor,
    DummyClassifier,
    DESSplitter,
)
from automl_models.components.estimators.ensemble.stack import DeepStackMixin

matthews_corrcoef_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
specificity_scorer = make_scorer(recall_score, pos_label=0)

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
        tuning_time: int = 600,
        target_metric=None,
        secondary_tuning_strategy: Optional[EnsembleStrategy] = None,
        main_stacking_ensemble: Optional[StackingEnsembleCreator] = None,
        secondary_ensembles: Optional[List[EnsembleCreator]] = None,
        stacking_level: int = 0,
        return_test_scores_during_tuning: bool = True,
        early_stopping: bool = False,
        cache: Union[str, bool] = False,
        random_state=None,
        secondary_level: Optional[ComponentLevel] = None,
        tune_kwargs: dict = None,
    ) -> None:
        self.problem_type = problem_type
        self.cv = cv
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.level = level
        self.tuner = tuner
        self.tuning_time = tuning_time
        self.target_metric = target_metric or self.default_metric_name
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.cache = cache
        self.secondary_tuning_strategy = (
            secondary_tuning_strategy
            or RoundRobinEstimator(configurations_to_select=10, percentile_threshold=15)
        )
        self.stacking_level = stacking_level
        self.main_stacking_ensemble = main_stacking_ensemble or StackingEnsembleCreator(
            ensemble_strategy=RoundRobinEstimator(
                configurations_to_select=10, percentile_threshold=15
            ),
            problem_type=self.problem_type,
        )
        self.secondary_ensembles = (
            secondary_ensembles
            or [
                # VotingEnsembleCreator(
                #     ensemble_strategy=RoundRobinEstimator(
                #         configurations_to_select=50, percentile_threshold=15
                #     ),
                #     problem_type=self.problem_type,
                # ),
                # VotingByMetricEnsembleCreator(
                #     ensemble_strategy=RoundRobinEstimator(
                #         configurations_to_select=50, percentile_threshold=15
                #     ),
                #     problem_type=self.problem_type,
                # ),
                SelectFromModelStackingEnsembleCreator(
                    ensemble_strategy=EnsembleBest(
                        configurations_to_select=100, percentile_threshold=1
                    ),
                    problem_type=self.problem_type,
                ),
            ]
            + [
                # VotingSoftEnsembleCreator(
                #     ensemble_strategy=RoundRobinEstimator(
                #         configurations_to_select=50, percentile_threshold=15
                #     ),
                #     problem_type=self.problem_type,
                # ),
                # VotingSoftByMetricEnsembleCreator(
                #     ensemble_strategy=RoundRobinEstimator(
                #         configurations_to_select=50, percentile_threshold=15
                #     ),
                #     problem_type=self.problem_type,
                # ),
            ]
            if self.problem_type.is_classification()
            else []
        )
        self.secondary_level = secondary_level
        self.tune_kwargs = tune_kwargs or {}

        self.secondary_tuner = None
        self.return_test_scores_during_tuning = return_test_scores_during_tuning

        self._displays = {
            "tuner_plot_display": IPythonDisplay("tuner_plot_display"),
        }

    def get_main_stacking_ensemble_at_level(
        self, stacking_level
    ):
        if stacking_level < 0:
            return None
        ret = self.ensembles_[stacking_level][self.main_stacking_ensemble_name]
        return ret

    @property
    def previous_main_stacking_ensemble(self):
        return self.get_main_stacking_ensemble_at_level(self.current_stacking_level - 1)

    @property
    def main_stacking_ensemble_name(self) -> str:
        return self.main_stacking_ensemble._ensemble_name

    @property
    def last_tuner_(self) -> Tuner:
        return self.tuners_[-1]

    @property
    def first_tuner_(self) -> Tuner:
        return self.tuners_[0]

    @property
    def scoring_dict(self) -> dict:
        # TODO fault tolerant metrics
        if self.problem_type == ProblemType.BINARY:
            return {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "roc_auc": "roc_auc",
                "precision": "precision",
                "recall": "recall",
                "specificity": specificity_scorer,
                "f1": "f1",
                "matthews_corrcoef": matthews_corrcoef_scorer,
            }
        elif self.problem_type == ProblemType.MULTICLASS:
            return {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "roc_auc": "roc_auc_ovr_weighted",
                "roc_auc_unweighted": "roc_auc_ovr",
                "precision_macro": "precision_macro",
                "precision_weighted": "precision_weighted",
                "recall_macro": "recall_macro",
                "recall_weighted": "recall_weighted",
                "f1_macro": "f1_macro",
                "f1_weighted": "f1_weighted",
                "matthews_corrcoef": matthews_corrcoef_scorer,
            }
        elif self.problem_type == ProblemType.REGRESSION:
            return {
                "r2": "r2",
                "neg_mean_absolute_error": "neg_mean_absolute_error",
                "neg_mean_squared_error": "neg_mean_squared_error",
                "neg_root_mean_squared_error": "neg_root_mean_squared_error",
                "neg_median_absolute_error": "neg_median_absolute_error",
            }

    @property
    def default_metric_name(self) -> str:
        if self.problem_type == ProblemType.BINARY:
            return "optimized_precision"
        elif self.problem_type == ProblemType.MULTICLASS:
            return "matthews_corrcoef"
        elif self.problem_type == ProblemType.REGRESSION:
            return "r2"

    def _get_cv(self, problem_type: ProblemType, cv: Union[BaseCrossValidator, int]):
        validate_type(cv, "cv", (BaseCrossValidator, _RepeatedSplits, int, None))
        return get_cv_for_problem_type(problem_type, n_splits=cv)

    def _tune(self, X, y, X_test=None, y_test=None, groups=None):
        categorical_columns = list(X.select_dtypes(include="category").columns)
        numeric_columns = list(X.select_dtypes(exclude="category").columns)
        pipeline_blueprint = create_pipeline_blueprint(
            problem_type=self.problem_type,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            level=self.level,
            X=X,
            y=y,
        )
        self.pipeline_blueprints_.append(pipeline_blueprint)
        if self.secondary_level:
            secondary_pipeline_blueprint = create_pipeline_blueprint(
                problem_type=self.problem_type,
                categorical_columns=categorical_columns,
                numeric_columns=numeric_columns,
                level=self.secondary_level,
                X=X,
                y=y,
                is_secondary=True,
            )
            self.secondary_pipeline_blueprints_.append(secondary_pipeline_blueprint)
        else:
            secondary_pipeline_blueprint = None

        tuner = self.tuner(
            problem_type=self.problem_type,
            pipeline_blueprint=pipeline_blueprint,
            secondary_pipeline_blueprint=secondary_pipeline_blueprint,
            random_state=self.random_state,
            cv=deepcopy(self.cv_),
            early_stopping=self.early_stopping,
            cache=self.cache,
            time_budget_s=self.tuning_time // (self.stacking_level + 1),
            scoring=self.scoring_dict,
            target_metric=self.target_metric,
            display=self._displays["tuner_plot_display"],
            **self.tune_kwargs,
        )
        self.tuners_.append(tuner)
        gc.collect()

        logger.debug("starting tuning", flush=True)
        self.last_tuner_.fit(X, y, X_test=X_test, y_test=y_test, groups=groups)
        logger.debug("tuning complete", flush=True)

        trial_ids_to_remove = set()
        results = self.last_tuner_.analysis_.results
        for trial_id, result in results.items():
            if not result.get("done", False) or pd.isnull(
                result.get("mean_validation_score", None)
            ):
                trial_ids_to_remove.add(trial_id)
                logger.debug(f"removing {trial_id}")
                continue
            if "config" in result:
                est = self._create_estimator(
                    result["config"],
                    pipeline_blueprint=pipeline_blueprint,
                    cache=self.last_tuner_._cache,  # TODO make dynamic
                )
                result["estimator"] = est
        results = {k: v for k, v in results.items() if k not in trial_ids_to_remove}
        results_df = self.last_tuner_.analysis_.results_df
        results_df = results_df[results_df.index.isin(set(results))]
        self.all_results_.append(results)
        self.all_results_df_.append(results_df)
        return results, results_df, pipeline_blueprint

    def _fit_one_layer(
        self,
        X,
        y,
        X_test=None,
        y_test=None,
        X_test_original=None,
        y_test_original=None,
        X_original=None,
        y_original=None,
        groups=None,
    ):
        self.current_stacking_level += 1
        logger.debug(f"current_stacking_level: {self.current_stacking_level}")
        logger.debug(X.columns)

        if self.return_test_scores_during_tuning:
            X_test_tuning = X_test
            y_test_tuning = y_test
        else:
            X_test_tuning = None
            y_test_tuning = None
        results, results_df, pipeline_blueprint = self._tune(
            X, y, X_test=X_test_tuning, y_test=y_test_tuning, groups=groups
        )

        # if "dataset_fraction" in results_df.columns:
        #     results_df = results_df[
        #         results_df["dataset_fraction"] >= results_df["dataset_fraction"].max()
        #     ]  # TODO make dynamic, add a warning if 100% of resource is not reached

        if self.secondary_tuner is not None:
            self._run_secondary_tuning(X, y, pipeline_blueprint, groups=groups)

        gc.collect()

        # with joblib.parallel_backend("ray"):
        X_stack, X_test_stack = self._create_ensembles(
            X_original,
            y_original,
            results,
            results_df,
            pipeline_blueprint,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
        )
        if self.current_stacking_level >= self.stacking_level:
            # logger.debug("fitting final ensemble", flush=True)
            # self.final_ensemble_ = self._create_final_stack()
            # self.final_ensemble_ = self._create_dynamic_ensemble(X, y)
            return
        self.meta_columns_.extend(X_stack.columns)
        multiprocessing_cache.clear()
        return self._fit_one_layer(
            pd.concat((X, X_stack), axis=1),
            y,
            X_test=pd.concat((X_test, X_test_stack), axis=1)
            if X_test is not None
            else X_test,
            y_test=y_test,
            X_test_original=X_test_original,
            y_test_original=y_test_original,
            X_original=X_original,
            y_original=y_original,
            groups=groups,
        )

    def _create_ensembles(
        self,
        X,
        y,
        results,
        results_df,
        pipeline_blueprint,
        X_test=None,
        y_test=None,
        X_test_original=None,
        y_test_original=None,
        save_to_ray: bool = True,
    ):
        ensemble_config = {
            "X": X,
            "y": y,
            "results": results,
            "results_df": results_df,
            "pipeline_blueprint": pipeline_blueprint,
            "metric_name": self.default_metric_name,
            # TODO fix this
            "metric": self.scoring_dict[self.target_metric]
            if self.problem_type == ProblemType.REGRESSION
            else self.scoring_dict["balanced_accuracy"],
            "random_state": self.random_state,
            "current_stacking_level": self.current_stacking_level,
            "previous_stack": self.get_main_stacking_ensemble_at_level(
                self.current_stacking_level - 1
            ),
            "X_test": X_test,
            "y_test": y_test,
            "X_test_original": X_test_original,
            "y_test_original": y_test_original,
            "cv": deepcopy(self.cv_),
            "cache": self.last_tuner_._cache,  # TODO make dynamic
        }

        print("creating ensembles")

        # cpus_available = ray.available_resources()["CPU"]

        ray_ensemble_config = ensemble_config
        ray_scoring_dict = self.scoring_dict
        ray_jobs = [
            ray_fit_ensemble_and_return_stacked_preds_remote(
                self.main_stacking_ensemble,
                ray_ensemble_config,
                ray_scoring_dict,
                #ray_cache_actor=self.last_tuner_.ray_cache_,
            )
        ]
        ray_jobs += [
            ray_fit_ensemble(
                ensemble,
                ray_ensemble_config,
                ray_scoring_dict,
                #ray_cache_actor=self.last_tuner_.ray_cache_,
            )
            for ensemble in self.secondary_ensembles or []
        ]
        print("starting ensemble jobs")

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        ray_results = ray_jobs

        print("ensemble jobs returned")

        def index_of_first(lst, pred):
            for i, v in enumerate(lst):
                if pred(v):
                    return i
            return None

        main_result = ray_results.pop(
            index_of_first(ray_results, lambda x: len(x) > 2)
        )
        (
            main_stacking_ensemble_fitted,
            X_stack,
            X_test_stack,
            score
        ) = main_result
        scores = {self.main_stacking_ensemble._ensemble_name: score}
        if ray_results:
            scores = {
                **scores,
                **{
                    ensemble._ensemble_name: ray_results[i][1]
                    for i, ensemble in enumerate(self.secondary_ensembles or [])
                },
            }
            fitted_ensembles = {
                ensemble._ensemble_name: ray_results[i][0]
                for i, ensemble in enumerate(self.secondary_ensembles or [])
            }
        else:
            fitted_ensembles = {}
        fitted_ensembles[
            self.main_stacking_ensemble._ensemble_name
        ] = main_stacking_ensemble_fitted

        self.ensemble_results_.append({})
        self.ensembles_.append(fitted_ensembles)

        for ensemble_name, score in scores.items():
            self._score_ensemble(ensemble_name, score)

        return X_stack, X_test_stack

    def _score_ensemble(
        self,
        ensemble_name,
        scores,
    ):
        if self.problem_type == ProblemType.BINARY:
            scores["optimized_precision"] = optimized_precision(
                scores["accuracy"], scores["recall"], scores["specificity"]
            )
        self.ensemble_results_[-1][ensemble_name] = scores

    def _create_dynamic_ensemble(self, X, y):
        # TODO fix
        des = DESSplitter(
            [est for name, est in self.ensembles_[-1].estimators],
            METADES(DFP=True),
            random_state=self.random_state,
            n_jobs=-1,
        )
        des.fit(X, y)
        return des

    def _create_final_stack(self):
        # TODO change isinstance
        if len(self.ensembles_) <= 1:
            return self.get_main_stacking_ensemble_at_level(-1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            cloned_ensembles = [
                self.get_main_stacking_ensemble_at_level(i)
                for i, _ in enumerate(self.ensembles_)
            ]
        for idx in range(0, len(cloned_ensembles) - 1):
            cloned_ensembles[idx].passthrough = True
            cloned_ensembles[idx].final_estimator = cloned_ensembles[idx + 1]
            cloned_ensembles[idx].final_estimator_ = cloned_ensembles[idx + 1]
        return cloned_ensembles[0]

    def _run_secondary_tuning(self, X, y, pipeline_blueprint, groups=None):
        raise NotImplementedError()
        groupby_list = [
            f"config{DELIM}{k}" for k in pipeline_blueprint.get_all_distributions().keys()
        ]
        groupby_list.reverse()
        grouped_results_df = self.all_results_df_[-1].groupby(by=groupby_list)
        for name, group in grouped_results_df:
            if group["mean_validation_score"].max() >= percentile:
                group_trial_ids = set(group.index)
                known_points = [
                    (
                        {
                            config_key: config_value
                            for config_key, config_value in v["config"].items()
                            if config_value != "passthrough"
                        },
                        v["mean_validation_score"],
                    )
                    for k, v in self.all_results_[-1].items()
                    if k in group_trial_ids
                ]
                logger.debug(known_points[0])
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
                secondary_results_df = secondary_results_df.dropna(
                    subset=["done"], how="any"
                )
                secondary_results_df = secondary_results_df[
                    secondary_results_df["done"]
                ]
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

    def _stack_estimator_if_needed(self, estimator, stacking_level):
        if isinstance(estimator, DeepStackMixin):
            return estimator
        stack = self.get_main_stacking_ensemble_at_level(stacking_level)
        return stack_estimator(estimator, stack)

    def _create_estimator(self, config, pipeline_blueprint, cache=False):
        default_config = {
            k: "passthrough"
            for k, v in pipeline_blueprint.get_all_distributions(
                use_extended=True
            ).items()
        }
        config = {**default_config, **config}
        config.pop("dataset_fraction", None)
        estimator = pipeline_blueprint(random_state=self.random_state)
        config_called = treat_config(
            config,
            self.last_tuner_.component_strings_,
            self.last_tuner_.hyperparameter_names_,
            self.random_state,
        )
        if "maxBins" in config_called:
            raise ValueError(str(estimator) + str(config_called))
        estimator.set_params(**config_called)
        if cache:
            memory = dynamic_memory_factory(cache)  # TODO make dynamic
            estimator.set_params(memory=memory)
        return estimator

    def get_ensemble_by_id(self, ensemble_id):
        check_is_fitted(self)
        stacking_level, ensemble_name = ensemble_id.split("_")
        ret = self.ensembles_[int(stacking_level)][ensemble_name]
        return ret

    def get_pipeline_by_id(self, id):
        check_is_fitted(self)
        for stacking_level, stacking_results in enumerate(self.all_results_):
            result = stacking_results.get(id, None)
            if result:
                return self._stack_estimator_if_needed(
                    estimator=result["estimator"], stacking_level=stacking_level - 1
                )
        if not result:
            raise KeyError(f"id {id} not present in results")

    def get_ensemble_or_pipeline_by_id(self, id):
        try:
            return self.get_ensemble_by_id(id)
        except (KeyError, ValueError):
            return self.get_pipeline_by_id(id)

    def fit(self, X, y, X_test=None, y_test=None, groups=None):
        self.current_stacking_level = -1

        self.cv_ = self._get_cv(self.problem_type, self.cv)
        self.pipeline_blueprints_: List[TopPipeline] = []
        self.secondary_pipeline_blueprints_: List[TopPipeline] = []
        self.tuners_: List[Tuner] = []
        self.secondary_tuners_ = defaultdict(list)
        self.all_results_ = []
        self.all_results_df_ = []
        self.ensembles_ = []
        self.ensemble_results_ = []
        self.meta_columns_ = []

        self._fit_one_layer(
            X,
            y,
            X_test=X_test,
            y_test=y_test,
            X_test_original=X_test,
            y_test_original=y_test,
            X_original=X,
            y_original=y,
            groups=groups,
        )
        return self
