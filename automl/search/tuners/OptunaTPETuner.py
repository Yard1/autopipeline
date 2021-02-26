from time import time
from typing import Optional, Union, Dict, List, Tuple
from collections import defaultdict

import pickle
from copy import copy

import numpy as np
import pandas as pd

from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

import optuna as ot
from optuna.samplers import BaseSampler
import optuna.distributions
from optuna.trial import TrialState

from .tuner import RayTuneTuner
from .utils import get_conditions, get_all_tunable_params
from ..distributions import CategoricalDistribution, get_optuna_trial_suggestions
from ...problems import ProblemType
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage
from ...utils.string import removeprefix

import logging

logger = logging.getLogger(__name__)


class ConditionalOptunaSearch(OptunaSearch):
    def __init__(
        self,
        space: Optional[Union[Dict, List[Tuple]]] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        keep_const_values: bool = True,
    ):
        assert ot is not None, "Optuna must be installed! Run `pip install optuna`."
        super(OptunaSearch, self).__init__(
            metric=metric, mode=mode, max_concurrent=None, use_early_stopped_trials=None
        )

        self._conditional_space = get_conditions(space, to_str=True)
        space, _ = get_all_tunable_params(space, to_str=True)
        self._const_values = {
            k: v.values[0]
            for k, v in space.items() if isinstance(v, CategoricalDistribution) and len(v.values) == 1
        }
        space = {
            k: v for k, v in space.items() if k not in self._const_values
        }
        self._space = get_optuna_trial_suggestions(space)

        if not keep_const_values:
            self._const_values = {}

        self._conditional_space = {k: v for k, v in self._conditional_space.items() if k in self._space}

        self._points_to_evaluate = points_to_evaluate
        if self._points_to_evaluate:
            n_startup_trials = max(10-len(self._points_to_evaluate), 4)
        else:
            n_startup_trials = 6

        self._study_name = "optuna"  # Fixed study name for in-memory storage
        self._sampler = sampler or ot.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            seed=seed,
        )
        assert isinstance(self._sampler, BaseSampler), (
            "You can only pass an instance of `optuna.samplers.BaseSampler` "
            "as a sampler to `OptunaSearcher`."
        )

        self._pruner = ot.pruners.NopPruner()
        self._storage = ot.storages.InMemoryStorage()

        self._ot_trials = {}
        self._ot_study = None
        if self._space:
            self._setup_study(mode)

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
            self._const_values
        )
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
        (
            self._storage,
            self._pruner,
            self._sampler,
            self._ot_trials,
            self._ot_study,
            self._points_to_evaluate,
            self._conditional_space,
            self._space,
            self._const_values
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

    def _get_params(self, ot_trial):
        params_checked = set()
        params = {}

        for key, condition in self._conditional_space.items():
            if key not in params:
                value = self._get_optuna_trial_value(ot_trial, self._space[key])
                params[key] = value
                params_checked.add(key)
            else:
                value = params[key]
            for dependent_name, required_values in condition.items():
                if dependent_name not in self._space:
                    continue
                params_checked.add(dependent_name)
                if value in required_values:
                    params[dependent_name] = self._get_optuna_trial_value(
                        ot_trial, self._space[dependent_name]
                    )

        for key, tpl in self._space.items():
            if key in params_checked:
                continue
            value = self._get_optuna_trial_value(ot_trial, self._space[key])
            params[key] = value

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
                values = ConditionalOptunaSearch.get_categorical_values(args, kwargs, offset=0)
                if len(values) <= 1:
                    continue
                distributions[k] = optuna.distributions.CategoricalDistribution(
                    *args, **kwargs
                )
            else:
                raise ValueError(f"Unknown distribution suggester {fn}")
        return distributions

    def add_evaluated_trial(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False,
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
            return False

        print("evaluated trial result")
        print(result)

        if not result:
            return False

        config = {
            removeprefix(k, "config/"): v
            for k, v in result.items()
            if k.startswith("config/")
        }
        distributions = {k: v for k, v in self._space.items() if k in config}
        distributions = ConditionalOptunaSearch.convert_optuna_params_to_distributions(
            distributions
        )
        config = {k: v for k, v in config.items() if k in distributions}

        trial = ot.trial.create_trial(
            state=TrialState.COMPLETE,
            value=result.get(self.metric, None),
            params=config,
            distributions=distributions,
        )
        self._ot_trials[trial_id] = trial

        len_studies = len(self._ot_study.trials)
        self._ot_study.add_trial(trial)
        assert len_studies < len(self._ot_study.trials)

        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
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

        print(f"Optuna has {len(self._ot_study.trials)} trials in memory")

        if trial_id not in self._ot_trials:
            ot_trial_id = self._storage.create_new_trial(self._ot_study._study_id)
            self._ot_trials[trial_id] = ot.trial.Trial(self._ot_study, ot_trial_id)
        ot_trial = self._ot_trials[trial_id]

        if self._points_to_evaluate:
            params = self._points_to_evaluate.pop(0)
        else:
            # getattr will fetch the trial.suggest_ function on Optuna trials
            params = self._get_params(ot_trial)
        params = {**self._const_values, **params}
        return unflatten_dict(params)


class OptunaTPETuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        num_samples: int = 100,
        early_stopping=True,
        early_stopping_brackets=1,
        cache=False,
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
            **tune_kwargs,
        )

    def _set_up_early_stopping(self, X, y, groups=None):
        if self.early_stopping:
            min_dist = self.cv.get_n_splits(self.X_, self.y_, self.groups_) * 2
            if self.problem_type.is_classification():
                min_dist *= len(self.y_.cat.categories)
            min_dist /= self.X_.shape[0]

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
            assert (
                self.early_stopping_fractions_[0] < self.early_stopping_fractions_[1]
            ), f"Could not generate correct fractions for the given number of splits. {self.early_stopping_fractions_}"
        else:
            self.early_stopping_fractions_ = [1]
        print(self.early_stopping_fractions_)
        self._tune_kwargs["scheduler"] = (
            ASHAScheduler(
                metric="mean_test_score",
                mode="max",
                reduction_factor=reduction_factor,
                max_t=self.early_stopping_splits_,
                brackets=self.early_stopping_brackets,
            )
            if self.early_stopping
            else None
        )

    def _pre_search(self, X, y, groups=None):
        super()._pre_search(X, y, groups=groups)
        _, self._component_strings_ = get_all_tunable_params(self.pipeline_blueprint)
        for conf in self.default_grid:
            for k, v in conf.items():
                if str(v) in self._component_strings_:
                    conf[k] = str(v)
        self._tune_kwargs["search_alg"] = ConditionalOptunaSearch(
            space=self.pipeline_blueprint,
            metric="mean_test_score",
            mode="max",
            points_to_evaluate=self.default_grid,
            seed=self.random_state,
        )

    def _treat_config(self, config):
        config = {k: self._component_strings_.get(v, v) for k, v in config.items()}
        return super()._treat_config(config)

    def _search(self, X, y, groups=None):
        self._pre_search(X, y, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, groups=None):
        return self._search(X, y, groups=groups)