from sklearn.utils.validation import check_is_fitted
from automl.utils.display import IPythonDisplay
from sklearn.base import clone
from typing import Optional, Union, Tuple

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

register_ray()


from ..ensemble import EnsembleStrategy, RoundRobin, RoundRobinEstimator
from ..utils import ray_context, optimized_precision
from ..tuners.tuner import Tuner
from ..tuners.OptunaTPETuner import OptunaTPETuner
from ..tuners.blendsearch import BlendSearchTuner
from ..tuners.BOHBTuner import BOHBTuner
from ..tuners.HEBOTuner import HEBOTuner
from ..tuners.utils import treat_config
from ..blueprints.pipeline import create_pipeline_blueprint
from ..cv import get_cv_for_problem_type
from ...components.component import ComponentLevel, ComponentConfig
from ...components.estimators.ensemble.stack import (
    PandasStackingClassifier,
    PandasStackingRegressor,
)
from ...components.estimators.ensemble.voting import (
    PandasVotingClassifier,
    DummyClassifier,
    PandasVotingRegressor,
)
from ...components.estimators.ensemble.des import DESSplitter, METADES, KNORAE
from ...components.estimators.linear_model import LogisticRegressionCV, ElasticNetCV
from ...problems.problem_type import ProblemType
from ...search.tuners.trainable import SklearnTrainable
from ...utils import validate_type
from ...utils.memory import dynamic_memory_factory

from sklearn.metrics import matthews_corrcoef, recall_score, make_scorer

matthews_corrcoef_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
specificity_scorer = make_scorer(recall_score, pos_label=0)

import logging

logger = logging.getLogger(__name__)

STACK_NAME = "Stack"


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
        best_percentile: int = 15,  # TODO: Move to EnsembleStrategy
        max_stacking_size: int = 10,  # TODO: Move to EnsembleStrategy
        stacking_strategy: Optional[EnsembleStrategy] = None,
        stacking_level: int = 0,
        max_voting_size: int = 100,  # TODO: Move to EnsembleStrategy
        voting_strategy: Optional[EnsembleStrategy] = None,
        return_test_scores_during_tuning: bool = True,
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
        self.tuning_time = tuning_time
        self.target_metric = target_metric or self.default_metric_name
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.cache = cache
        self.best_percentile = best_percentile
        self.max_stacking_size = max_stacking_size
        self.stacking_strategy = stacking_strategy or RoundRobinEstimator()
        self.stacking_level = stacking_level
        self.max_voting_size = max_voting_size
        self.voting_strategy = voting_strategy or RoundRobinEstimator()
        self.tune_kwargs = tune_kwargs or {}

        self.secondary_tuner = None
        self.return_test_scores_during_tuning = return_test_scores_during_tuning

        self._displays = {
            "tuner_plot_display": IPythonDisplay("tuner_plot_display"),
        }

    @property
    def last_tuner_(self):
        return self.tuners_[-1]

    @property
    def first_tuner_(self):
        return self.tuners_[0]

    @property
    def scoring_dict(self):
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
    def default_metric_name(self):
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
            time_budget_s=self.tuning_time // (self.stacking_level + 1),
            scoring=self.scoring_dict,
            target_metric=self.target_metric,
            display=self._displays["tuner_plot_display"],
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
                if trial_id in self.last_tuner_.fitted_estimators_:
                    result["estimator"] = self.last_tuner_.fitted_estimators_[trial_id]
                else:
                    result["estimator"] = self._create_estimator(
                        result["config"],
                        pipeline_blueprint=pipeline_blueprint,
                        cache=False,  # TODO make dynamic
                    )
        results = {k: v for k, v in results.items() if k not in trial_ids_to_remove}
        results_df = self.last_tuner_.analysis_.results_df
        results_df = results_df[results_df.index.isin(set(results))]
        self.all_results_.append(results)
        self.all_results_df_.append(results_df)
        return results, results_df, pipeline_blueprint

    def _fit_one_layer(self, X, y, X_test=None, y_test=None, groups=None):
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

        if "dataset_fraction" in results_df.columns:
            results_df = results_df[
                results_df["dataset_fraction"] >= results_df["dataset_fraction"].max()
            ]  # TODO make dynamic, add a warning if 100% of resource is not reached
        percentile = self.best_percentile

        if self.secondary_tuner is not None:
            self._run_secondary_tuning(
                X, y, pipeline_blueprint, percentile, groups=groups
            )

        self.ensemble_results_.append({})
        self.ensembles_.append({})
        self._create_voting_by_metric_ensemble(
            X,
            y,
            results,
            results_df,
            percentile,
            pipeline_blueprint,
            X_test=X_test,
            y_test=y_test,
        )
        self._create_voting_ensemble(
            X,
            y,
            results,
            results_df,
            percentile,
            pipeline_blueprint,
            X_test=X_test,
            y_test=y_test,
        )
        X_stack, X_test_stack = self._create_stacking_ensemble(
            X,
            y,
            results,
            results_df,
            percentile,
            pipeline_blueprint,
            X_test=X_test,
            y_test=y_test,
        )
        del self.last_tuner_.fold_predictions_
        if self.current_stacking_level >= self.stacking_level:
            logger.debug("fitting final ensemble", flush=True)
            self.final_ensemble_ = self._create_final_stack()
            # self.final_ensemble_ = self._create_dynamic_ensemble(X, y)
            return
        self.meta_columns_.extend(X_stack.columns)
        return self._fit_one_layer(
            pd.concat((X, X_stack), axis=1),
            y,
            X_test=pd.concat((X_test, X_test_stack), axis=1)
            if X_test is not None
            else X_test,
            y_test=y_test,
            groups=groups,
        )

    def _create_stacking_ensemble(
        self,
        X,
        y,
        results,
        results_df,
        percentile,
        pipeline_blueprint,
        X_test=None,
        y_test=None,
        final_estimator=None,
    ):
        """
        Creates a classic stacking ensemble
        """
        ensemble_name = STACK_NAME
        if self.problem_type == ProblemType.REGRESSION:
            metric = self.scoring_dict[self.target_metric]
            final_estimator = (
                final_estimator
                or ElasticNetCV(
                    random_state=self.random_state,
                    cv=KFold(shuffle=True, random_state=self.random_state),
                )()
            )
            ensemble_class = PandasStackingRegressor
        elif self.problem_type.is_classification():
            metric = self.scoring_dict["balanced_accuracy"]  # TODO fix after adding op as a scorer
            final_estimator = (
                final_estimator
                or LogisticRegressionCV(
                    scoring=metric,
                    random_state=self.random_state,
                    cv=StratifiedKFold(shuffle=True, random_state=self.random_state),
                )()
            )
            ensemble_class = PandasStackingClassifier
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

        trial_ids_for_ensembling = self._select_trial_ids_for_stacking(
            results, results_df, pipeline_blueprint, percentile
        )
        trials_for_ensembling = [results[k] for k in trial_ids_for_ensembling]
        estimators = self._get_estimators_for_ensemble(trials_for_ensembling)
        if not estimators:
            raise ValueError("No estimators selected for stacking!")
        logger.debug(f"creating stacking classifier with {len(estimators)}")
        ensemble = ensemble_class(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.cv_,
            n_jobs=None,
        )
        logger.debug("ensemble created")
        gc.collect()
        logger.debug("fitting ensemble")
        print("fitting ensemble")
        ensemble.n_jobs = -1  # TODO make dynamic
        ensemble.fit(
            X,
            y,
            predictions=[
                self.last_tuner_.fold_predictions_[k] for k in trial_ids_for_ensembling
            ],
            refit_estimators=False,
        )
        X_stack = ensemble.stacked_predictions_
        self.ensembles_[-1][ensemble_name] = ensemble
        self._score_ensemble(
            ensemble, ensemble_name, X, y, X_test=X_test, y_test=y_test
        )
        ensemble.n_jobs = None
        if X_test is not None:
            X_test_stack = ensemble.transform(X_test)
        else:
            X_test_stack = None

        return X_stack, X_test_stack

    def _get_optimal_voting_ensemble_weights(
        self,
        trial_results,
        initial_guess,
        y_test,
        target_metric=None,
    ):
        """
        Optimize on CV predictions
        """
        target_metric = target_metric or self.default_metric_name
        initial_guess = np.array(initial_guess)
        sum_initial_guess = np.sum(initial_guess)
        initial_guess = (initial_guess / sum_initial_guess)[:-1]
        estimators = [
            (
                str(idx),
                DummyClassifier(
                    str(idx),
                    self.last_tuner_.fold_predictions_[trial_result["trial_id"]][
                        "predict"
                    ],
                    self.last_tuner_.fold_predictions_[trial_result["trial_id"]][
                        "predict_proba"
                    ],
                ),
            )
            for idx, trial_result in enumerate(trial_results)
        ]
        le_ = LabelEncoder().fit(y_test)
        y_test = le_.transform(y_test)
        ensemble = PandasVotingClassifier(
            estimators=estimators,
            voting="hard",
            n_jobs=1,
            weights=initial_guess,
        )
        ensemble.le_ = le_
        ensemble.classes_ = ensemble.le_.classes_
        ensemble.estimators_ = [x[1] for x in estimators]

        def optimize_function(weights, *args):
            weights = np.append(weights, 1 - np.sum(weights))
            logger.debug(np.min(weights))
            ensemble.weights = weights
            scores, _ = SklearnTrainable.score_test(
                ensemble,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                y_test,
                self.scoring_dict,
                refit=False,
                error_score="raise",
            )
            # logger.debug(-1 * scores[target_metric])
            return -1 * scores[target_metric]

        constraints = {"type": "ineq", "fun": lambda x: 1 - np.sum(x)}
        logger.debug(initial_guess)
        bounds = tuple((0, 1) for x in initial_guess)
        res = scipy.optimize.minimize(
            optimize_function,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={
                "maxiter": 1000,
                "disp": True,
            },
        )
        return np.append(res.x, 1 - np.sum(res.x))

    def _create_voting_by_metric_ensemble(
        self,
        X,
        y,
        results,
        results_df,
        percentile,
        pipeline_blueprint,
        X_test=None,
        y_test=None,
    ):
        """
        Creates a voting ensemble.
        """
        ensemble_name = "VotingByMetric"
        # TODO support user defined metrics better
        metric_name = self.default_metric_name
        if self.problem_type == ProblemType.REGRESSION:
            weight_function = (
                lambda trial: trial["metrics"][metric_name]
                if trial["metrics"][metric_name] > 0.5
                else 0
            )
            ensemble_class = PandasVotingRegressor
            ensemble_args = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                weight_function = (
                    lambda trial: trial["metrics"][metric_name]
                    if trial["metrics"][metric_name] > 0.5
                    else 0
                )
            else:
                weight_function = (
                    lambda trial: trial["metrics"][metric_name]
                    if trial["metrics"][metric_name] > 0
                    else 0
                )
            ensemble_class = PandasVotingClassifier
            ensemble_args = {"voting": "hard"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

        trial_ids_for_ensembling = self._select_trial_ids_for_voting(
            results, results_df, pipeline_blueprint, percentile
        )
        trials_for_ensembling = [results[k] for k in trial_ids_for_ensembling]
        logger.debug(f"creating voting classifier with {self.max_voting_size}")
        weights = [weight_function(trial) for trial in trials_for_ensembling]
        trials_for_ensembling = [
            results[k]
            for idx, k in enumerate(trial_ids_for_ensembling)
            if weights[idx] > 0
        ]
        weights = [weight for weight in weights if weight > 0]
        estimators = self._get_estimators_for_ensemble(trials_for_ensembling)
        if not estimators:
            raise ValueError("No estimators selected for ensembling!")
        logger.debug(f"final number of estimators: {len(estimators)}")
        ensemble = ensemble_class(
            estimators=estimators,
            n_jobs=None,
            weights=weights,
            **ensemble_args,
        )
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
        self.ensembles_[-1][ensemble_name] = ensemble
        self._score_ensemble(
            ensemble, ensemble_name, X, y, X_test=X_test, y_test=y_test
        )
        ensemble.n_jobs = None

    def _create_voting_ensemble(
        self,
        X,
        y,
        results,
        results_df,
        percentile,
        pipeline_blueprint,
        X_test=None,
        y_test=None,
    ):
        """
        Creates a voting ensemble.
        """
        ensemble_name = "VotingUniform"
        metric_name = self.default_metric_name
        if self.problem_type == ProblemType.REGRESSION:
            weight_function = (
                lambda trial: 1
                if trial["metrics"][metric_name] > 0.5
                else 0
            )
            ensemble_class = PandasVotingRegressor
            ensemble_args = {}
        elif self.problem_type.is_classification():
            if self.problem_type == ProblemType.BINARY:
                weight_function = (
                    lambda trial: 1
                    if trial["metrics"][metric_name] > 0.5
                    else 0
                )
            else:
                weight_function = (
                    lambda trial: 1
                    if trial["metrics"][metric_name] > 0
                    else 0
                )
            ensemble_class = PandasVotingClassifier
            ensemble_args = {"voting": "hard"}
        else:
            raise ValueError(f"Unknown ProblemType {self.problem_type}")

        trial_ids_for_ensembling = self._select_trial_ids_for_voting(
            results, results_df, pipeline_blueprint, percentile
        )
        trials_for_ensembling = [results[k] for k in trial_ids_for_ensembling]
        logger.debug(f"creating voting classifier with {self.max_voting_size}")
        weights = [weight_function(trial) for trial in trials_for_ensembling]
        trials_for_ensembling = [
            results[k]
            for idx, k in enumerate(trial_ids_for_ensembling)
            if weights[idx] > 0
        ]
        estimators = self._get_estimators_for_ensemble(trials_for_ensembling)
        if not estimators:
            raise ValueError("No estimators selected for ensembling!")
        logger.debug(f"final number of estimators: {len(estimators)}")
        ensemble = ensemble_class(
            estimators=estimators,
            n_jobs=None,
            **ensemble_args,
        )
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
        self.ensembles_[-1][ensemble_name] = ensemble
        self._score_ensemble(
            ensemble, ensemble_name, X, y, X_test=X_test, y_test=y_test
        )
        ensemble.n_jobs = None

    def _get_estimators_for_ensemble(self, trials_for_ensembling):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return [
                (
                    f"meta-{self.current_stacking_level}_{trial_result['trial_id']}",
                    deepcopy(trial_result["estimator"]),
                )
                for trial_result in trials_for_ensembling
            ]

    def _score_ensemble(self, ensemble, ensemble_name, X, y, X_test=None, y_test=None):
        if X_test is None:
            return
        logger.debug("scoring ensemble")
        scoring = self.scoring_dict
        scores, _ = SklearnTrainable.score_test(
            ensemble,
            X,
            y,
            X_test,
            y_test,
            scoring,
            refit=False,
            error_score=np.nan,
        )
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
            return self.ensembles_[-1][STACK_NAME]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            cloned_ensembles = [
                deepcopy(ensembles[STACK_NAME]) for ensembles in self.ensembles_
            ]
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
            config, self.last_tuner_._component_strings_, self.random_state
        )
        estimator.set_params(**config_called)
        if cache:
            estimator.memory = dynamic_memory_factory(cache)  # TODO make dynamic
        return estimator

    # TODO refactor those, use objects
    def _select_trial_ids_for_stacking(
        self, results: dict, results_df: pd.DataFrame, pipeline_blueprint, percentile
    ) -> list:
        return self.stacking_strategy.select_trial_ids(
            results=results,
            results_df=results_df,
            configurations_to_select=self.max_stacking_size,
            pipeline_blueprint=pipeline_blueprint,
            percentile_threshold=percentile,
        )

    def _select_trial_ids_for_voting(
        self, results: dict, results_df: pd.DataFrame, pipeline_blueprint, percentile
    ) -> list:
        return self.voting_strategy.select_trial_ids(
            results=results,
            results_df=results_df,
            configurations_to_select=self.max_voting_size,
            pipeline_blueprint=pipeline_blueprint,
            percentile_threshold=percentile,
        )

    def get_ensemble_by_id(self, ensemble_id):
        check_is_fitted(self)
        stacking_level, ensemble_name = ensemble_id.split("_")
        return self.ensembles_[stacking_level][ensemble_name]

    def fit(self, X, y, X_test=None, y_test=None, groups=None):
        self.current_stacking_level = -1

        self.cv_ = self._get_cv(self.problem_type, self.cv)
        self.pipeline_blueprints_ = []
        self.tuners_ = []
        self.secondary_tuners_ = defaultdict(list)
        self.all_results_ = []
        self.all_results_df_ = []
        self.ensembles_ = []
        self.ensemble_results_ = []
        self.meta_columns_ = []

        with ray_context(
            global_checkpoint_s=self.tune_kwargs.pop("TUNE_GLOBAL_CHECKPOINT_S", 10)
        ), joblib.parallel_backend("ray"):
            self._fit_one_layer(X, y, X_test=X_test, y_test=y_test, groups=groups)
        return self