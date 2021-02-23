from time import time
from typing import Optional, Union, Dict, List, Tuple

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

from .tuner import RayTuneTuner
from ..distributions import CategoricalDistribution
from ...problems import ProblemType
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage

import logging

logger = logging.getLogger(__name__)


def get_optuna_trial_suggestion(ot_trial, value, label, use_default: bool = False):
    if use_default:
        return value.default
    if isinstance(value, CategoricalDistribution) and len(value.values) <= 1:
        return value.values[0]
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
        n_startup_trials = 10

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
                name = v.get_hyperparameter_key_suffix(k, k2)
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
        num_samples: int = 500,
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
        self._tune_kwargs["search_alg"] = ConditionalOptunaSearch(
            space=self.pipeline_blueprint,
            metric="mean_test_score",
            mode="max",
            points_to_evaluate=self.default_grid,
            seed=self.random_state,
        )

    def _search(self, X, y, groups=None):
        self._pre_search(X, y, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, groups=None):
        return self._search(X, y, groups=groups)