from time import time
from typing import Optional, Union, Dict, List, Tuple
from collections import defaultdict

import pickle
import warnings
from copy import copy

import numpy as np
import pandas as pd

from ray import cloudpickle
from ray.tune.result import DEFAULT_METRIC, TRAINING_ITERATION
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

import optuna as ot
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler, RandomSampler
import optuna.distributions
from optuna.trial import TrialState

from automl.search.samplers.optuna_rf_sampler import CatBoostSampler

from .tuner import RayTuneTuner
from .utils import get_conditions, get_all_tunable_params
from ..distributions import CategoricalDistribution, get_optuna_trial_suggestions
from ...problems import ProblemType
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage
from ...utils.string import removeprefix
from ...utils.display import IPythonDisplay

import logging

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", (UserWarning, ExperimentalWarning, FutureWarning))


class NoRelativeParamsTrial(ot.trial.Trial):
    def _init_relative_params(self) -> None:

        trial = self.storage.get_trial(self._trial_id)

        study = ot.pruners._filter_study(self.study, trial)

        self.relative_search_space = self.study.sampler.infer_relative_search_space(
            study, trial
        )
        self.relative_params = {}


def add_dynamic_trial_to_study(study: ot.Study, trial: ot.trial.FixedTrial):
    # Sync storage once every trial.
    study._storage.read_trials_from_remote_storage(study._study_id)

    trial_id = study._pop_waiting_trial_id()
    if trial_id is None:
        trial_id = study._storage.create_new_trial(
            study._study_id, template_trial=trial
        )
    trial = NoRelativeParamsTrial(study, trial_id)

    return trial


class ConditionalOptunaSearch(OptunaSearch):
    def __init__(
        self,
        space: Optional[Union[Dict, List[Tuple]]] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        n_startup_trials: Optional[int] = None,
        use_extended: bool = False,
        remove_const_values: bool = False,
        **kwargs,
    ):
        assert ot is not None, "Optuna must be installed! Run `pip install optuna`."
        super(OptunaSearch, self).__init__(
            metric=metric, mode=mode
        )

        self._conditional_space = get_conditions(
            space, to_str=True, use_extended=use_extended
        )
        space, _, _ = get_all_tunable_params(
            space, to_str=True, use_extended=use_extended
        )
        if remove_const_values:
            const_values = {
                k
                for k, v in space.items()
                if isinstance(v, CategoricalDistribution) and len(v.values) == 1
            }
            space = {k: v for k, v in space.items() if k not in const_values}
        self._space = space
        self._ot_space = get_optuna_trial_suggestions(space)

        self._points_to_evaluate = points_to_evaluate or []
        assert n_startup_trials is None or isinstance(n_startup_trials, int)
        if n_startup_trials is None:
            n_startup_trials = max(len(space), 10)
        #     if self._points_to_evaluate:
        #         n_startup_trials = max(10 - len(self._points_to_evaluate), 10)
        #     else:
        #         n_startup_trials = 10

        self._study_name = "optuna"  # Fixed study name for in-memory storage
        self._sampler = sampler or self._get_sampler(n_startup_trials, seed, **kwargs)
        assert isinstance(self._sampler, BaseSampler), (
            "You can only pass an instance of `optuna.samplers.BaseSampler` "
            "as a sampler to `OptunaSearcher`."
        )

        self._ot_trials = {}
        self._ot_study: Optional[ot.Study] = None
        if self._space:
            self._setup_study(mode)

    def _get_sampler(self, n_startup_trials, seed, **kwargs):
        params = dict(
            multivariate=True,
            group=True,
            constant_liar=True,
            n_ei_candidates=2048,
            warn_independent_sampling=False,
        )
        kwargs = {**params, **kwargs}
        return ot.samplers.TPESampler(
            n_startup_trials=n_startup_trials, seed=seed, **kwargs
        )

    def save(self, checkpoint_path: str):
        save_object = (
            self._storage,
            self._pruner,
            self._sampler,
            self._ot_trials,
            self._ot_study,
            self._points_to_evaluate,
            self._conditional_space,
            self._space,
            self._ot_space,
        )
        with open(checkpoint_path, "wb") as outputFile:
            cloudpickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = cloudpickle.load(inputFile)
        (
            self._storage,
            self._pruner,
            self._sampler,
            self._ot_trials,
            self._ot_study,
            self._points_to_evaluate,
            self._conditional_space,
            self._space,
            self._ot_space,
        ) = save_object

    @staticmethod
    def get_trial_suggestion_name(args, kwargs):
        return args[0] if len(args) > 0 else kwargs["name"]

    @staticmethod
    def get_categorical_values(args, kwargs, offset=1):
        return args[offset] if len(args) > offset else kwargs["choices"]

    def _get_optuna_trial_value(self, ot_trial, tpl):
        fn, args, kwargs = tpl
        if (
            ConditionalOptunaSearch.get_trial_suggestion_name(args, kwargs)
            == "suggest_categorical"
        ):
            values = ConditionalOptunaSearch.get_categorical_values(args, kwargs)
            if len(values) <= 1:
                return values[0]
        return getattr(ot_trial, fn)(*args, **kwargs)

    def _get_params(self, ot_trial: ot.Trial, space, params=None):
        params = params or {}

        for key, condition in self._conditional_space.items():
            if key not in params:
                value = self._get_optuna_trial_value(ot_trial, space[key])
                params[key] = value
            else:
                value = params[key]
            for dependent_name, required_values in condition[value].items():
                if dependent_name in params:
                    continue
                if dependent_name in space and (
                    required_values is True or len(required_values) > 1
                ):
                    try:
                        params[dependent_name] = self._get_optuna_trial_value(
                            ot_trial, space[dependent_name]
                        )
                    except ValueError:
                        params[dependent_name] = required_values[0]
                elif len(required_values) == 1:
                    params[dependent_name] = required_values[0]
                else:
                    params[dependent_name] = "passthrough"

        return params

    @staticmethod
    def convert_optuna_params_to_distributions(params):
        distributions = {}
        for k, v in params.items():
            fn, args, kwargs = v
            args = args[1:] if len(args) > 0 else args
            kwargs = kwargs.copy()
            kwargs.pop("name", None)
            if fn == "suggest_loguniform":
                distributions[k] = optuna.distributions.LogUniformDistribution(
                    *args, **kwargs
                )
            elif fn == "suggest_discrete_uniform":
                distributions[k] = optuna.distributions.DiscreteUniformDistribution(
                    *args, **kwargs
                )
            elif fn == "suggest_uniform":
                distributions[k] = optuna.distributions.UniformDistribution(
                    *args, **kwargs
                )
            elif fn == "suggest_int":
                if kwargs.pop("log", False) or args[-1] is True:
                    if args[-1] is True:
                        args = args[:-1]
                    distributions[k] = optuna.distributions.IntLogUniformDistribution(
                        *args, **kwargs
                    )
                else:
                    distributions[k] = optuna.distributions.IntUniformDistribution(
                        *args, **kwargs
                    )
            elif fn == "suggest_categorical":
                values = ConditionalOptunaSearch.get_categorical_values(
                    args, kwargs, offset=0
                )
                if len(values) <= 1:
                    continue
                distributions[k] = optuna.distributions.CategoricalDistribution(
                    *args, **kwargs
                )
            else:
                raise ValueError(f"Unknown distribution suggester {fn}")
        return distributions

    def on_trial_result(self, trial_id: str, result: Dict, step=None):
        metric = result[self.metric]
        step = result[TRAINING_ITERATION] if step is None else step
        ot_trial = self._ot_trials[trial_id]
        ot_trial.report(metric, step)

    def on_trial_complete(
        self,
        trial_id: str,
        result: Optional[Dict] = None,
        error: bool = False,
        state: TrialState = TrialState.COMPLETE,
        num_intermediate_values: int = 1,
    ):
        ot_trial = self._ot_trials[trial_id]

        if state == TrialState.PRUNED:
            for i in range(num_intermediate_values):
                self.on_trial_result(trial_id, result, step=i)
        if state != TrialState.COMPLETE:
            val = None
        else:
            val = result.get(self.metric, None) if result else None
        try:
            self._ot_study.tell(ot_trial, val, state=state)
            if state == TrialState.FAIL:
                self._ot_trials.pop(trial_id)
        except ValueError as exc:
            logger.warning(exc)  # E.g. if NaN was reported
            print(exc)

    def add_evaluated_trial(
        self,
        trial_id: str,
        result: Optional[Dict] = None,
        error: bool = False,
        state: TrialState = TrialState.COMPLETE,
        num_intermediate_values: int = 1,
    ):
        if not self._space:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"
                )
            )
        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__, metric=self._metric, mode=self._mode
                )
            )

        if trial_id in self._ot_trials:
            logger.debug(f"{trial_id} already in ot_trials")
            return False

        if not result:
            return False

        is_result = any(k.startswith("config/") for k in result)
        config = {
            removeprefix(k, "config/"): v
            for k, v in result.items()
            if (not is_result or k.startswith("config/")) and v != "passthrough"
        }
        distributions = {k: v for k, v in self._ot_space.items() if k in config}
        distributions = ConditionalOptunaSearch.convert_optuna_params_to_distributions(
            distributions
        )
        config = {k: v for k, v in config.items() if k in distributions}
        assert config
        trial = ot.trial.create_trial(
            state=state,
            value=result.get(self.metric, None),
            params=config,
            distributions=distributions,
            intermediate_values={
                k: result.get(self.metric, None) for k in range(num_intermediate_values)
            },
            system_attrs={"fixed_params": config},
        )

        len_studies = len(self._ot_study.trials)
        self._ot_trials[trial_id] = add_dynamic_trial_to_study(self._ot_study, trial)
        assert len_studies < len(self._ot_study.trials)

        return True

    def suggest(
        self, trial_id: str, reask: bool = False, params=None
    ) -> Optional[Dict]:
        if not self._space:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"
                )
            )
        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__, metric=self._metric, mode=self._mode
                )
            )

        if trial_id not in self._ot_trials:
            self._ot_trials[trial_id] = self._ot_study.ask()
        elif reask:
            self._ot_study.tell(self._ot_trials[trial_id], None, TrialState.FAIL)
            self._ot_trials[trial_id] = self._ot_study.ask()
        ot_trial = self._ot_trials[trial_id]

        params = self._get_params(ot_trial, self._ot_space, params=params)
        logger.debug(params)
        return unflatten_dict(params)


class ConditionalOptunaSearchCatBoost(ConditionalOptunaSearch):
    def _get_sampler(self, n_startup_trials, seed, **kwargs):
        params = dict(
            constant_liar=True,
            n_ei_candidates=20000,
            warn_independent_sampling=False,
            random_fraction=0.1,
        )
        kwargs = {**params, **kwargs}
        return CatBoostSampler(n_startup_trials=n_startup_trials, seed=seed, **kwargs)

    def suggest(
        self,
        trial_id: str,
        reask: bool = False,
        params=None,
        ei_space=None,
    ) -> Optional[Dict]:
        if not self._space:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"
                )
            )
        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__, metric=self._metric, mode=self._mode
                )
            )

        def ei_objective(trial):
            self._get_params(trial, ei_space or self._ot_space)

        self._ot_study.sampler._ei_objective = ei_objective

        if trial_id not in self._ot_trials:
            self._ot_trials[trial_id] = self._ot_study.ask()
        elif reask:
            self._ot_study.tell(self._ot_trials[trial_id], None, TrialState.FAIL)
            self._ot_trials[trial_id] = self._ot_study.ask()
        ot_trial = self._ot_trials[trial_id]
        params = self._get_params(ot_trial, self._ot_space, params=params)
        logger.debug(params)
        self._ot_study.sampler._ei_objective = None
        return unflatten_dict(params)


class OptunaTPETuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        num_samples: int = 100,
        secondary_pipeline_blueprint=None,
        early_stopping=True,
        early_stopping_brackets=1,
        cache=False,
        max_concurrent: int = 1,
        trainable_n_jobs: int = 4,
        display: Optional[IPythonDisplay] = None,
        **tune_kwargs,
    ) -> None:
        self.early_stopping = early_stopping
        self.early_stopping_brackets = early_stopping_brackets
        super().__init__(
            problem_type=problem_type,
            pipeline_blueprint=pipeline_blueprint,
            cv=cv,
            random_state=random_state,
            num_samples=num_samples,
            cache=cache,
            display=display,
            max_concurrent=max_concurrent,
            trainable_n_jobs=trainable_n_jobs,
            secondary_pipeline_blueprint=secondary_pipeline_blueprint,
            **tune_kwargs,
        )

    def _set_up_early_stopping(self, X, y, groups=None):
        if self.early_stopping and self.X_.shape[0] > 20000:
            min_dist = self.cv.get_n_splits(self.X_, self.y_, self.groups_) * 20
            if self.problem_type.is_classification():
                min_dist *= len(self.y_.cat.categories)
            min_dist /= self.X_.shape[0]
            min_dist = max(min_dist, 10000 / self.X_.shape[0])

            reduction_factor = 4
            self.early_stopping_splits_ = (
                -int(np.log(min_dist / 1.0) / np.log(reduction_factor)) + 1
            )
            self.early_stopping_fractions_ = 1.0 * np.power(
                reduction_factor,
                -np.linspace(
                    self.early_stopping_splits_ - 1, 0, self.early_stopping_splits_
                ),
            )
            if len(self.early_stopping_fractions_) > 1:
                assert (
                    self.early_stopping_fractions_[0]
                    < self.early_stopping_fractions_[1]
                ), f"Could not generate correct fractions for the given number of splits. {self.early_stopping_fractions_}"
            else:
                self.early_stopping_fractions_[0] = 1
        else:
            self.early_stopping_fractions_ = [1]
        logger.debug(self.early_stopping_fractions_)
        self._tune_kwargs["scheduler"] = (
            ASHAScheduler(
                metric="mean_validation_score",
                mode="max",
                reduction_factor=reduction_factor,
                max_t=self.early_stopping_splits_,
                brackets=self.early_stopping_brackets,
            )
            if len(self.early_stopping_fractions_) > 1
            else None
        )

    def _pre_search(self, X, y, X_test=None, y_test=None, groups=None):
        super()._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)
        self._tune_kwargs["search_alg"] = ConditionalOptunaSearch(
            space=self.pipeline_blueprint,
            metric="mean_validation_score",
            mode="max",
            points_to_evaluate=self.default_grid_,
            seed=self.random_state,
        )
        logger.debug(f"cache: {self._cache}")
