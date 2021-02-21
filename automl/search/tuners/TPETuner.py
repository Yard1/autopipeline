from time import time
from typing import Optional, Union, Dict, List, Tuple

from copy import copy

import numpy as np
import pandas as pd

from sklearn.model_selection._search import ParameterGrid

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

import optuna as ot
from optuna.samplers import BaseSampler

from .tuner import RayTuneTuner
from .utils import ray_context, split_list_into_chunks
from ...problems import ProblemType
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage

import logging

logger = logging.getLogger(__name__)


def get_optuna_trial_suggestion(ot_trial, value, label, use_default: bool = False):
    if use_default:
        return value.default
    fn, args, kwargs = value.get_optuna(label)
    return getattr(ot_trial, fn)(*args, **kwargs)


class ConditionalOptunaSearch(OptunaSearch):
    def __init__(
        self,
        space: Optional[Union[Dict, List[Tuple]]] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
    ):
        assert ot is not None, "Optuna must be installed! Run `pip install optuna`."
        super(OptunaSearch, self).__init__(
            metric=metric, mode=mode, max_concurrent=None, use_early_stopped_trials=None
        )

        self._space = space

        self._points_to_evaluate = points_to_evaluate
        n_startup_trials = min(10 - len(points_to_evaluate), 1)

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

    def _fetch_params(self, ot_trial, spec, use_default: bool = False):
        spec = copy(spec)
        config = {}
        estimator_name, estimators_distributon = spec.get_estimator_distribution()
        estimator = get_optuna_trial_suggestion(
            ot_trial, estimators_distributon, estimator_name, use_default=use_default
        )
        spec.remove_invalid_components(
            pipeline_config=ComponentConfig(estimator=estimator),
            current_stage=AutoMLStage.TUNE,
        )
        config[estimator_name] = estimator
        preprocessors_grid = spec.get_preprocessor_distribution()
        for k, v in preprocessors_grid.items():
            config[k] = get_optuna_trial_suggestion(
                ot_trial, v, k, use_default=use_default
            )

        hyperparams = {}

        for k, v in config.items():
            for k2, v2 in v.get_tuning_grid().items():
                name = f"{k}__{v.prefix}_{k2}"
                choice = get_optuna_trial_suggestion(
                    ot_trial, v2, name, use_default=use_default
                )
                hyperparams[name] = choice
            config[k] = v

        final_config = {**config, **hyperparams}
        return final_config

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

        if trial_id not in self._ot_trials:
            ot_trial_id = self._storage.create_new_trial(self._ot_study._study_id)
            self._ot_trials[trial_id] = ot.trial.Trial(self._ot_study, ot_trial_id)
        ot_trial = self._ot_trials[trial_id]

        if self._points_to_evaluate:
            params = self._points_to_evaluate.pop(0)
        else:
            # getattr will fetch the trial.suggest_ function on Optuna trials
            params = self._fetch_params(ot_trial, self._space)
        return unflatten_dict(params)


class OptunaTPETuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        early_stopping=True,
        early_stopping_splits=3,
        early_stopping_brackets=1,
    ) -> None:
        self.problem_type = problem_type
        self.pipeline_blueprint = pipeline_blueprint
        self.cv = cv
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.early_stopping_splits = early_stopping_splits
        self.early_stopping_brackets = early_stopping_brackets
        super().__init__()

    def _get_single_default_hyperparams(self, components):
        hyperparams = {}
        for k, v in components.items():
            for k2, v2 in v.get_tuning_grid().items():
                name = f"{k}__{v.prefix}_{k2}"
                hyperparams[name] = v2.default
        return {**components, **hyperparams}

    def _get_default_components(self, pipeline_blueprint) -> dict:
        default_grid = {
            k: v.values for k, v in pipeline_blueprint.get_all_distributions().items()
        }
        return [
            self._get_single_default_hyperparams(components)
            for components in ParameterGrid(default_grid)
        ]

    def _search(self, X, y):
        self.X_ = X
        self.y_ = y

        if self.early_stopping:
            min_dist = self.cv.get_n_splits(self.X_, self.y_) * 2
            if self.problem_type.is_classification():
                min_dist *= len(self.y_.cat.categories)
            min_dist /= self.X_.shape[0]

            # from https://github.com/automl/HpBandSter/blob/master/hpbandster/optimizers/bohb.py
            self.early_stopping_fractions_ = 1.0 * np.power(3, -np.linspace(self.early_stopping_splits-1, 0, self.early_stopping_splits))
            self.early_stopping_fractions_[0] = max(self.early_stopping_fractions_[0], min_dist)
        else:
            self.early_stopping_fractions_ = [1]

        default_grid = self._get_default_components(self.pipeline_blueprint)
        preset_configurations = [
            config
            for config in self.pipeline_blueprint.preset_configurations
            if config not in default_grid
        ]
        default_grid += preset_configurations

        scheluder = (
            ASHAScheduler(
                metric="mean_test_score",
                mode="max",
                reduction_factor=3,
                max_t=self.early_stopping_splits,
                brackets=self.early_stopping_brackets
            )
            if self.early_stopping
            else None
        )

        with ray_context():
            self.analysis_ = tune.run(
                self._trial_with_cv,
                search_alg=ConditionalOptunaSearch(
                    space=self.pipeline_blueprint,
                    metric="mean_test_score",
                    mode="max",
                    points_to_evaluate=default_grid,
                    seed=self.random_state,
                ),
                scheduler=scheluder,
                #metric="mean_test_score",
                #mode="max",
                #num_samples=10,
                num_samples=50 + len(default_grid),
                verbose=1,
                reuse_actors=True,
                fail_fast=True,
            )

        return self

    def fit(self, X, y):
        return self._search(X, y)