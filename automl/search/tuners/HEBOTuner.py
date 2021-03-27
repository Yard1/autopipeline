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
from ...utils.display import IPythonDisplay


import logging

logger = logging.getLogger(__name__)


class HEBOTuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        use_extended: bool = True,
        num_samples: int = 24,
        early_stopping=True,
        early_stopping_brackets=1,
        cache=False,
        display: Optional[IPythonDisplay] = None,
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
            use_extended=use_extended,
            display=display,
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
        logger.debug(self.early_stopping_fractions_)
        self._tune_kwargs["scheduler"] = (
            ASHAScheduler(
                metric="mean_validation_score",
                mode="max",
                reduction_factor=reduction_factor,
                max_t=self.early_stopping_splits_,
                brackets=self.early_stopping_brackets,
            )
            if self.early_stopping
            else None
        )

    def _pre_search(self, X, y, X_test=None, y_test=None, groups=None):
        logger.debug("_pre_search")
        points_to_evaluate, evaluated_rewards = zip(*self.known_points)
        logger.debug(points_to_evaluate[0])
        self.pipeline_blueprint = copy(self.pipeline_blueprint)
        super()._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)
        space = {}
        for k, v in self.pipeline_blueprint.get_all_distributions(
            use_extended=self.use_extended
        ).items():
            if k in points_to_evaluate[0] and points_to_evaluate[0][k] != "passthrough":
                space[k] = CategoricalDistribution(
                    [next(x for x in v.values if str(x) == points_to_evaluate[0][k])]
                )
        logger.debug(space)
        space, _ = get_all_tunable_params(
            self.pipeline_blueprint,
            to_str=True,
            use_extended=self.use_extended,
            space=space,
        )
        if "Estimator__\u200bRandomForestClassifier\u200b__n_estimators" in space:
            space[
                "Estimator__\u200bRandomForestClassifier\u200b__n_estimators"
            ].upper = max(
                points_to_evaluate,
                key=lambda x: x[
                    "Estimator__\u200bRandomForestClassifier\u200b__n_estimators"
                ],
            )[
                "Estimator__\u200bRandomForestClassifier\u200b__n_estimators"
            ]
        if "Estimator__\u200bLGBMClassifier\u200b__n_estimators" in space:
            space["Estimator__\u200bLGBMClassifier\u200b__n_estimators"].upper = max(
                points_to_evaluate,
                key=lambda x: x["Estimator__\u200bLGBMClassifier\u200b__n_estimators"],
            )["Estimator__\u200bLGBMClassifier\u200b__n_estimators"]
        if "Estimator__\u200bLGBMClassifier\u200b__num_leaves" in space:
            space["Estimator__\u200bLGBMClassifier\u200b__num_leaves"].upper = max(
                points_to_evaluate,
                key=lambda x: x["Estimator__\u200bLGBMClassifier\u200b__num_leaves"],
            )["Estimator__\u200bLGBMClassifier\u200b__num_leaves"]
        self._const_values = {
            k: v.values[0]
            for k, v in space.items()
            if (isinstance(v, CategoricalDistribution) and len(v.values) == 1)
            or k == "Estimator"
        }
        self._const_values["Estimator"] = points_to_evaluate[0]["Estimator"]
        space = {k: v for k, v in space.items() if k not in self._const_values}
        default_values = {
            k: v.default for k, v in space.items() if k not in points_to_evaluate[0]
        }
        logger.debug(space)
        space = get_tune_distributions(space)
        points_to_evaluate = [
            {**default_values, **{k: v for k, v in x.items() if k in space}}
            for x in points_to_evaluate
        ]
        logger.debug(points_to_evaluate)
        logger.debug(list(evaluated_rewards))
        self._tune_kwargs["search_alg"] = ConcurrencyLimiter(
            HEBOSearch(
                space=space,
                metric="mean_validation_score",
                mode="max",
                points_to_evaluate=points_to_evaluate,
                evaluated_rewards=list(evaluated_rewards),
                random_state_seed=self.random_state,
                n_suggestions=8,
            ),
            max_concurrent=8,
            batch=True,
        )