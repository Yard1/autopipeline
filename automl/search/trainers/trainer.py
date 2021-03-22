from sklearn.base import clone
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import gc

from copy import deepcopy
from collections import defaultdict

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import scipy.optimize

import joblib
import ray
from ray.util.joblib import register_ray

register_ray()


from ..ensemble import EnsembleStrategy, RoundRobin, RoundRobinEstimator
from ..utils import ray_context, f2_mcc_roc_auc
from ..tuners.tuner import Tuner
from ..tuners.OptunaTPETuner import OptunaTPETuner
from ..tuners.blendsearch import BlendSearchTuner
from ..tuners.BOHBTuner import BOHBTuner
from ..tuners.HEBOTuner import HEBOTuner
from ..tuners.utils import treat_config
from ..blueprints.pipeline import create_pipeline_blueprint
from ..cv import get_cv_for_problem_type
from ...components.component import ComponentLevel, ComponentConfig
from ...components.estimators.ensemble.stack import PandasStackingClassifier
from ...components.estimators.ensemble.voting import (
    PandasVotingClassifier,
    DummyClassifier,
)
from ...components.estimators.ensemble.des import DESSplitter, METADES, KNORAE
from ...components.estimators.linear_model import LogisticRegression
from ...problems.problem_type import ProblemType
from ...search.tuners.trainable import SklearnTrainable
from ...utils import validate_type
from ...utils.memory import dynamic_memory_factory

from sklearn.metrics import matthews_corrcoef, make_scorer

matthews_corrcoef_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)

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
        best_percentile: int = 15,  # TODO: Move to EnsembleStrategy
        max_stacking_size: int = 10,  # TODO: Move to EnsembleStrategy
        stacking_strategy: Optional[EnsembleStrategy] = None,
        stacking_level: int = 0,
        max_voting_size: int = -1,  # TODO: Move to EnsembleStrategy
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
        # TODO regression

    def _get_cv(self, problem_type: ProblemType, cv: Union[BaseCrossValidator, int]):
        validate_type(cv, "cv", (BaseCrossValidator, int, None))
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
        )
        self.tuners_.append(tuner)
        gc.collect()

        print("starting tuning", flush=True)
        self.last_tuner_.fit(X, y, X_test=X_test, y_test=y_test, groups=groups)
        print("tuning complete", flush=True)

        trial_ids_to_remove = set()
        results = self.last_tuner_.analysis_.results
        for trial_id, result in results.items():
            if not result.get("done", False) or pd.isnull(result.get("mean_validation_score", None)):
                trial_ids_to_remove.add(trial_id)
                print(f"removing {trial_id}")
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
        print(f"current_stacking_level: {self.current_stacking_level}")
        print(X.columns)

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
            print("fitting final ensemble", flush=True)
            self.final_ensemble_ = self._create_final_ensemble()
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
        trial_ids_for_ensembling = self._select_trial_ids_for_stacking(
            results, results_df, pipeline_blueprint, percentile
        )
        trials_for_ensembling = [results[k] for k in trial_ids_for_ensembling]
        estimators = self._get_estimators_for_ensemble(trials_for_ensembling)
        if not estimators:
            raise ValueError("No estimators selected for stacking!")
        final_estimator = final_estimator or LogisticRegression()()
        print(f"creating stacking classifier with {len(estimators)}")
        ensemble = PandasStackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.cv_,
            n_jobs=None,
        )
        print("ensemble created")
        gc.collect()
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
        self.ensembles_.append(ensemble)
        self._score_ensemble(ensemble, X, y, X_test=X_test, y_test=y_test)
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
        target_metric="f2_mcc_roc_auc",
    ):
        """
        Optimize on CV predictions
        """
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
            voting="soft",
            n_jobs=1,
            weights=initial_guess,
        )
        ensemble.le_ = le_
        ensemble.classes_ = ensemble.le_.classes_
        ensemble.estimators_ = [x[1] for x in estimators]

        def optimize_function(weights, *args):
            weights = np.append(weights, 1 - np.sum(weights))
            print(np.min(weights))
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
            # print(-1 * scores[target_metric])
            return -1 * scores[target_metric]

        constraints = {"type": "ineq", "fun": lambda x: 1 - np.sum(x)}
        print(initial_guess)
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
        Creates a voting ensemble using Gini weights as described in https://dmip.webs.upv.es/papers/AppliedInt.pdf
        @article{10.1007/s10489-012-0388-2,
        author = {Bella, Antonio and Ferri, C\`{e}sar and Hern\'{a}ndez-Orallo, Jos\'{e} and Ram\'{\i}rez-Quintana, Mar\'{\i}a Jos\'{e}},
        title = {On the Effect of Calibration in Classifier Combination},
        year = {2013},
        issue_date = {June      2013},
        publisher = {Kluwer Academic Publishers},
        address = {USA},
        volume = {38},
        number = {4},
        issn = {0924-669X},
        url = {https://doi.org/10.1007/s10489-012-0388-2},
        doi = {10.1007/s10489-012-0388-2},
        abstract = {A general approach to classifier combination considers each model as a probabilistic classifier which outputs a class membership posterior probability. In this general scenario, it is not only the quality and diversity of the models which are relevant, but the level of calibration of their estimated probabilities as well. In this paper, we study the role of calibration before and after classifier combination, focusing on evaluation measures such as MSE and AUC, which better account for good probability estimation than other evaluation measures. We present a series of findings that allow us to recommend several layouts for the use of calibration in classifier combination. We also empirically analyse a new non-monotonic calibration method that obtains better results for classifier combination than other monotonic calibration methods.},
        journal = {Applied Intelligence},
        month = jun,
        pages = {566–585},
        numpages = {20},
        keywords = {Classifier combination, Classifier diversity, Classifier calibration, Calibration measures, Separability measures, Probability estimation}
        }
        """
        trial_ids_for_ensembling = self._select_trial_ids_for_voting(
            results, results_df, pipeline_blueprint, percentile
        )
        trials_for_ensembling = [results[k] for k in trial_ids_for_ensembling]
        print(f"creating voting classifier with {self.max_voting_size}")
        weights = [
            max(0, (trial["metrics"]["roc_auc"] - 0.5) * 2)
            for trial in trials_for_ensembling
        ]
        trials_for_ensembling = [
            results[k]
            for idx, k in enumerate(trial_ids_for_ensembling)
            if weights[idx] > 0
        ]
        weights = [weight for weight in weights if weight > 0]
        estimators = self._get_estimators_for_ensemble(trials_for_ensembling)
        if not estimators:
            raise ValueError("No estimators selected for ensembling!")
        print(f"final number of estimators: {len(estimators)}")
        ensemble = PandasVotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=None,
            weights=weights,
        )
        print("ensemble created")
        gc.collect()
        print("fitting ensemble")
        ensemble.n_jobs = -1  # TODO make dynamic
        ensemble.fit(
            X,
            y,
            refit_estimators=False,
        )
        self.ensembles_.append(ensemble)
        self._score_ensemble(ensemble, X, y, X_test=X_test, y_test=y_test)
        ensemble.n_jobs = None

    def _get_estimators_for_ensemble(self, trials_for_ensembling):
        return [
            (
                f"meta-{self.current_stacking_level}_{trial_result['trial_id']}",
                deepcopy(trial_result["estimator"]),
            )
            for trial_result in trials_for_ensembling
        ]

    def _score_ensemble(self, ensemble, X, y, X_test=None, y_test=None):
        if X_test is None:
            return
        print("scoring ensemble")
        self.ensemble_results_[-1][str(ensemble)], _ = SklearnTrainable.score_test(
            ensemble, X, y, X_test, y_test, self.scoring_dict, refit=False, error_score="raise"
        )

    def _create_dynamic_ensemble(self, X, y):
        des = DESSplitter(
            [est for name, est in self.ensembles_[-1].estimators],
            METADES(DFP=True),
            random_state=self.random_state,
            n_jobs=-1,
        )
        des.fit(X, y)
        return des

    def _create_final_ensemble(self):
        # TODO change isinstance
        if len(self.ensembles_) <= 1:
            return next(
                ensemble
                for ensemble in self.ensembles_
                if isinstance(ensemble, PandasStackingClassifier)
            )
        cloned_ensembles = [
            deepcopy(ensemble)
            for ensemble in self.ensembles_
            if isinstance(ensemble, PandasStackingClassifier)
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
            return self._fit_one_layer(
                X, y, X_test=X_test, y_test=y_test, groups=groups
            )

        return self
