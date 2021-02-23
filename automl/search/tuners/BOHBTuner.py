from time import time
from typing import Optional, Union, Dict, List, Tuple

from copy import copy

import numpy as np
import pandas as pd

from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .tuner import RayTuneTuner
from ..distributions import CategoricalDistribution
from ...problems import ProblemType
from ...components.component import Component, ComponentConfig
from ...search.stage import AutoMLStage

import logging

logger = logging.getLogger(__name__)


class ConditionalTuneBOHB(TuneBOHB):
    def __init__(
        self,
        space: Optional[Union[Dict, CS.ConfigurationSpace]] = None,
        bohb_config: Optional[Dict] = None,
        max_concurrent: int = 10,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        seed: Optional[int] = None,
    ):
        from hpbandster.optimizers.config_generators.bohb import BOHB

        assert (
            BOHB is not None
        ), """HpBandSter must be installed!
            You can install HpBandSter with the command:
            `pip install hpbandster ConfigSpace`."""
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."

        self._seed = seed
        self._space = ConditionalTuneBOHB.convert_search_space(space, seed=self._seed)

        self._points_to_evaluate = points_to_evaluate
        n_startup_trials = 10

        self._bohb_config = bohb_config
        self._max_concurrent = max_concurrent
        self.trial_to_params = {}
        self.running = set()
        self.paused = set()
        self._metric = metric

        super(TuneBOHB, self).__init__(metric=self._metric, mode=mode)

        if self._space:
            self._setup_bohb()

    @staticmethod
    def convert_search_space(spec, seed=None, use_default: bool = False):
        cs = CS.ConfigurationSpace(seed=seed)
        spec = copy(spec)
        estimator_name, estimators = spec.get_estimator_distribution()
        estimators_distributon = estimators.get_CS(estimator_name)
        cs.add_hyperparameter(estimators_distributon)

        for estimator in estimators.values:
            for k2, v2 in estimator.get_tuning_grid().items():
                name = estimator.get_hyperparameter_key_suffix(estimator_name, k2)
                hyperparam_dist = v2.get_CS(name)
                cond = CS.EqualsCondition(
                    hyperparam_dist, estimators_distributon, estimator
                )
                cs.add_hyperparameter(hyperparam_dist)
                cs.add_condition(cond)

        preprocessors_grid = spec.get_preprocessor_distribution()
        print(preprocessors_grid)
        for k, v in preprocessors_grid.items():
            dist = v.get_CS(k)
            cs.add_hyperparameter(dist)
            for choice in v.values:
                for k2, v2 in choice.get_tuning_grid().items():
                    name = choice.get_hyperparameter_key_suffix(k, k2)
                    hyperparam_dist = v2.get_CS(name)
                    cond = CS.EqualsCondition(hyperparam_dist, dist, choice)
                    cs.add_hyperparameter(hyperparam_dist)
                    cs.add_condition(cond)

        last_estimator_idx = 0
        conditions = set()
        for estimator in estimators.values:
            spec_copy = copy(spec)
            spec_copy.remove_invalid_components(
                pipeline_config=ComponentConfig(estimator=estimator),
                current_stage=AutoMLStage.TUNE,
            )

            grid = spec_copy.get_preprocessor_distribution()
            # print(grid)
            removed_components = {
                k: [v2 for v2 in v.values if v2 not in grid[k].values]
                if k in grid
                else v.values
                for k, v in preprocessors_grid.items()
            }
            for k, v in removed_components.items():
                for v2 in v:
                    conditions.add(
                        (
                            estimators_distributon,
                            estimator,
                            cs.get_hyperparameter(k),
                            v2,
                        )
                    )
        print(len(conditions))
        for est_name, est, param_name, param in conditions:
            cond = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(est_name, est),
                CS.ForbiddenEqualsClause(param_name, param),
            )
            while True:
                try:
                    cs.add_forbidden_clause(cond)
                    break
                except:
                    cs.forbidden_clauses.pop()
                    last_estimator_idx += 1
                    estimators_distributon.default_value = estimators.values[
                        last_estimator_idx
                    ]

        return cs

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

        if len(self.running) < self._max_concurrent:
            if self._points_to_evaluate:
                config = self._points_to_evaluate.pop(0)
            else:
                # This parameter is not used in hpbandster implementation.
                config, info = self.bohber.get_config(None)
            self.trial_to_params[trial_id] = copy.deepcopy(config)
            self.running.add(trial_id)
            return unflatten_dict(config)
        return None


class BOHBTuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        early_stopping=True,
        early_stopping_splits=3,
        early_stopping_brackets=1,
        **tune_kwargs,
    ) -> None:
        self.early_stopping = early_stopping
        self.early_stopping_splits = early_stopping_splits
        self.early_stopping_brackets = early_stopping_brackets
        super().__init__(
            problem_type=problem_type,
            pipeline_blueprint=pipeline_blueprint,
            cv=cv,
            random_state=random_state,
            **tune_kwargs,
        )

    def _set_up_early_stopping(self, X, y, groups=None):
        pass

    def _pre_search(self, X, y, groups=None):
        super()._pre_search(X, y, groups=groups)
        self._tune_kwargs["search_alg"] = ConditionalTuneBOHB(
            space=self.pipeline_blueprint,
            metric="mean_test_score",
            mode="max",
            # points_to_evaluate=self.default_grid,
            seed=self.random_state,
        )

    def _search(self, X, y, groups=None):
        self._pre_search(X, y, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, groups=None):
        return self._search(X, y, groups=groups)