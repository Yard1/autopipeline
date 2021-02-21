from time import time
from typing import Optional, Union, Dict, List, Tuple

from copy import copy

import numpy as np
import pandas as pd

from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.model_selection._search import ParameterGrid

from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

import optuna as ot
from optuna.samplers import BaseSampler

from sklearn.base import clone
from sklearn.model_selection import cross_validate

from .tuner import Tuner
from .utils import ray_context
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage
from ..utils import call_component_if_needed

import logging

logger = logging.getLogger(__name__)


def get_optuna_dist(ot_trial, value, label, use_default: bool = False):
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
    ):
        assert ot is not None, "Optuna must be installed! Run `pip install optuna`."
        super(OptunaSearch, self).__init__(
            metric=metric, mode=mode, max_concurrent=None, use_early_stopped_trials=None
        )

        self._space = space

        self._points_to_evaluate = points_to_evaluate

        self._study_name = "optuna"  # Fixed study name for in-memory storage
        self._sampler = sampler or ot.samplers.TPESampler()
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
        estimator = get_optuna_dist(
            ot_trial, estimators_distributon, estimator_name, use_default=use_default
        )
        spec.remove_invalid_components(
            pipeline_config=ComponentConfig(estimator=estimator),
            current_stage=AutoMLStage.TUNE,
        )
        config[estimator_name] = estimator
        preprocessors_grid = spec.get_preprocessor_distribution()
        for k, v in preprocessors_grid.items():
            config[k] = get_optuna_dist(ot_trial, v, k, use_default=use_default)

        hyperparams = {}

        for k, v in config.items():
            for k2, v2 in v.get_tuning_grid().items():
                name = f"{k}__{v.prefix}_{k2}"
                choice = get_optuna_dist(ot_trial, v2, name, use_default=use_default)
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
            hyperparams = {}
            for k, v in params.items():
                for k2, v2 in v.get_tuning_grid().items():
                    name = f"{k}__{v.prefix}_{k2}"
                    hyperparams[name] = v2.default
            params = {**params, **hyperparams}
        else:
            # getattr will fetch the trial.suggest_ function on Optuna trials
            params = self._fetch_params(ot_trial, self._space)
        return unflatten_dict(params)


class TPETuner(Tuner):
    def __init__(self, pipeline_blueprint, random_state) -> None:
        self.pipeline_blueprint = pipeline_blueprint
        self.random_state = random_state
        super().__init__()

    def _trial_with_cv(self, config):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config_called = {
            k: call_component_if_needed(
                v, random_state=self.random_state, return_prefix_mixin=True
            )
            for k, v in config.items()
        }

        estimator.set_params(**config_called)

        scores = cross_validate(
            estimator,
            self.X_,
            self.y_,
            # cv=self.cv,
            # error_score=self.error_score,
            # fit_params=self.fit_params,
            # groups=self.groups,
            # return_train_score=self.return_train_score,
            # scoring=self.scoring,
        )

        tune.report(mean_test_score=np.mean(scores["test_score"]))

    def _search(self, X, y):
        self.X_ = X
        self.y_ = y
        default_grid = {
            k: v.values
            for k, v in self.pipeline_blueprint.get_all_distributions().items()
        }
        default_grid = list(ParameterGrid(default_grid))

        time_start = time()
        with ray_context():
            analysis = tune.run(
                self._trial_with_cv,
                search_alg=ConditionalOptunaSearch(
                    space=self.pipeline_blueprint,
                    metric="mean_test_score",
                    mode="max",
                    points_to_evaluate=default_grid,
                ),
                # num_samples = 10,
                num_samples=50 + len(default_grid),
                verbose=1,
                reuse_actors=True,
                fail_fast=True,
            )

    def fit(self, X, y):
        return self._search(X, y)