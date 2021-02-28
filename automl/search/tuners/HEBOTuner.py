from automl.search.distributions.distributions import get_tune_distributions
from time import time
from typing import Iterable, Optional, Union, Dict, List, Tuple
from collections import defaultdict

import pickle
from copy import copy

import numpy as np
import pandas as pd

from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from .HEBOSearch import HEBOSearch
from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

from .tuner import RayTuneTuner
from .utils import get_all_tunable_params
from ..distributions import CategoricalDistribution, get_optuna_trial_suggestions
from ...problems import ProblemType
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage
from ...utils.string import removeprefix
from ...utils.exceptions import validate_type

import logging

logger = logging.getLogger(__name__)


class HEBOTuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        num_samples: int = 24,
        early_stopping=True,
        early_stopping_brackets=1,
        cache=False,
        known_points=None,
        **tune_kwargs,
    ) -> None:
        validate_type(known_points, "known_points", Iterable)
        self.early_stopping = early_stopping
        self.early_stopping_brackets = early_stopping_brackets
        self.known_points = known_points
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
        self.pipeline_blueprint = copy(self.pipeline_blueprint)
        self.pipeline_blueprint.remove_invalid_components(
            pipeline_config=ComponentConfig(
                estimator=self.known_points[0][0]["Estimator"]
            ),
            current_stage=AutoMLStage.TUNE,
        )
        self.pipeline_blueprint.components[-1] = (self.pipeline_blueprint.components[-1][0], [next(
            x
            for x in self.pipeline_blueprint.components[-1][1]
            if str(x) == self.known_points[0][0]["Estimator"]
        )])
        super()._pre_search(X, y, groups=groups)
        self._tune_kwargs["num_samples"] -= len(self.default_grid)
        space, _ = get_all_tunable_params(
            self.pipeline_blueprint, to_str=True, use_extended=True
        )
        self._const_values = {
            k: v.values[0]
            for k, v in space.items()
            if (isinstance(v, CategoricalDistribution) and len(v.values) == 1)
            or k == "Estimator"
        }
        self._const_values["Estimator"] = self.known_points[0][0]["Estimator"]
        space = {k: v for k, v in space.items() if k not in self._const_values}
        default_values = {
            k: v.default for k, v in space.items() if k not in self.known_points[0][0]
        }
        space = get_tune_distributions(space)
        points_to_evaluate, evaluated_rewards = zip(*self.known_points)
        points_to_evaluate = [
            {**default_values, **{k: v for k, v in x.items() if k in space}}
            for x in points_to_evaluate
        ]
        self._tune_kwargs["search_alg"] = ConcurrencyLimiter(
            HEBOSearch(
                space=space,
                metric="mean_test_score",
                mode="max",
                points_to_evaluate=points_to_evaluate,
                evaluated_rewards=list(evaluated_rewards),
                random_state_seed=self.random_state,
                n_suggestions=8,
            ),
            max_concurrent=8,
            batch=True,
        )

    def _treat_config(self, config):
        config = {**self._const_values, **config}
        return super()._treat_config(config)

    def _search(self, X, y, groups=None):
        self._pre_search(X, y, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, groups=None):
        return self._search(X, y, groups=groups)