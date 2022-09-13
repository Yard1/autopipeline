"""!
 * Copyright (c) 2020-2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
"""
import collections
from copy import deepcopy
import pandas as pd
from typing import Dict, Optional, List, Tuple, Callable
import numpy as np
import traceback
import pickle
import time
from ray.tune.sample import Categorical

from sklearn.utils import Bunch

from ray import tune
from ray import cloudpickle
from ray.tune.suggest import Searcher, ConcurrencyLimiter
from ray.tune.utils.util import flatten_dict
from flaml.searcher.search_thread import SearchThread
from flaml.searcher.blendsearch import BlendSearch
from flaml.searcher.flow2 import FLOW2

import logging

from automl.search.distributions.distributions import get_optuna_trial_suggestions
from automl.search.utils import numpy_to_python

logger = logging.getLogger(__name__)

from ray.tune import sample

from .OptunaTPETuner import (
    ConditionalOptunaSearch,
    ConditionalOptunaSearchCatBoost,
    TrialState,
    RandomSampler,
)
from ..distributions import get_tune_distributions, CategoricalDistribution
from .utils import get_conditions, enforce_conditions_on_config, get_all_tunable_params
from .tuner import RayTuneTuner
from ...utils.tune_callbacks import BestPlotCallback, META_KEY
from ...problems import ProblemType
from ...utils.display import IPythonDisplay

EPS = 1e-6
GlobalSearch = ConditionalOptunaSearchCatBoost


# def print(*args, **kwargs):
#    pass


# TODO: Fix cost_attr in cache
class PatchedFLOW2(FLOW2):
    # STEPSIZE = 0.15

    def __init__(
        self,
        init_config: dict,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        cat_hp_cost: Optional[dict] = None,
        space: Optional[dict] = None,
        prune_attr: Optional[str] = None,
        min_resource: Optional[float] = None,
        max_resource: Optional[float] = None,
        resource_multiple_factor: Optional[float] = 4,
        seed: Optional[int] = 20,
        limit_space_to_init_config: bool = False,
        conditional_space=None,
        cost_attr="time_total_s",
        tol=0.001,
        hyperparameter_names=None,
    ):
        """Constructor

        Args:
            init_config: a dictionary from a subset of controlled dimensions
                to the initial low-cost values. e.g. {'epochs':1}
            metric: A string of the metric name to optimize for.
                minimization or maximization.
            mode: A string in ['min', 'max'] to specify the objective as
            cat_hp_cost: A dictionary from a subset of categorical dimensions
                to the relative cost of each choice.
                e.g.,

                .. code-block:: python

                    {'tree_method': [1, 1, 2]}

                i.e., the relative cost of the
                three choices of 'tree_method' is 1, 1 and 2 respectively.
            space: A dictionary to specify the search space.
            prune_attr: A string of the attribute used for pruning.
                Not necessarily in space.
                When prune_attr is in space, it is a hyperparameter, e.g.,
                    'n_iters', and the best value is unknown.
                When prune_attr is not in space, it is a resource dimension,
                    e.g., 'sample_size', and the peak performance is assumed
                    to be at the max_resource.
            min_resource: A float of the minimal resource to use for the
                prune_attr; only valid if prune_attr is not in space.
            max_resource: A float of the maximal resource to use for the
                prune_attr; only valid if prune_attr is not in space.
            resource_multiple_factor: A float of the multiplicative factor
                used for increasing resource.
            seed: An integer of the random seed.
        """
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
        else:
            mode = "min"

        super(FLOW2, self).__init__(metric=metric, mode=mode)
        # internally minimizes, so "max" => -1
        if mode == "max":
            self.metric_op = -1.0
        elif mode == "min":
            self.metric_op = 1.0
        self.space = space or {}
        self.conditional_space = conditional_space or {}
        self.signature_space = list(space.keys())
        self._random = np.random.RandomState(seed)
        self._seed = seed
        self._hyperparameter_names = hyperparameter_names or {}
        if not init_config:
            logger.warning(
                "No init config given to FLOW2. Using random initial config."
                "For cost-frugal search, "
                "consider providing init values for cost-related hps via "
                "'init_config'."
            )
        elif self.conditional_space:
            init_config = enforce_conditions_on_config(init_config, conditional_space)
            logger.debug(f"FLOW2 init_config {init_config}")
            print(f"FLOW2 init_config {init_config}")
        self.init_config = self.best_config = init_config
        if limit_space_to_init_config:
            assert init_config
            self.space = {
                k: v if k in self._hyperparameter_names else init_config[k]
                for k, v in self.space.items()
                if k in init_config
            }
            self.space["Estimator"] = init_config["Estimator"]
        self.cat_hp_cost = cat_hp_cost
        self.prune_attr = prune_attr
        self.min_resource = min_resource
        self.resource_multiple_factor = resource_multiple_factor or 4
        self.max_resource = max_resource
        self._resource = None
        self._step_lb = np.Inf
        self.saved_resource = None
        self.cost_attr = cost_attr
        if space:
            self._init_search()
        if self.prune_attr and self.prune_attr not in self.space and self.max_resource:
            self.signature_space.append(prune_attr)
        self._tol = tol

    # copied from FLAML==0.8.0
    def _init_search(self):
        self._tunable_keys = []
        self._bounded_keys = []
        self._ordered_choice_hp = {}
        self._ordered_cat_hp = {}
        self._unordered_cat_hp = {}
        self._cat_hp_cost = {}
        for key, domain in self.space.items():
            assert not (
                isinstance(domain, dict) and "grid_search" in domain
            ), f"{key}'s domain is grid search, not supported in FLOW^2."
            if callable(getattr(domain, "get_sampler", None)):
                self._tunable_keys.append(key)
                sampler = domain.get_sampler()
                # the step size lower bound for uniform variables doesn't depend
                # on the current config
                if isinstance(sampler, sample.Quantized):
                    q = sampler.q
                    sampler = sampler.get_sampler()
                    if str(sampler) == "Uniform":
                        self._step_lb = min(
                            self._step_lb, q / (domain.upper - domain.lower + 1)
                        )
                elif isinstance(domain, sample.Integer) and str(sampler) == "Uniform":
                    self._step_lb = min(
                        self._step_lb, 1.0 / (domain.upper - domain.lower)
                    )
                if isinstance(domain, sample.Categorical):
                    cat_hp_cost = self.cat_hp_cost
                    if cat_hp_cost and key in cat_hp_cost:
                        cost = np.array(cat_hp_cost[key])
                        ind = np.argsort(cost)
                        ordered = np.array(domain.categories)[ind]
                        cost = self._cat_hp_cost[key] = cost[ind]
                        d = {}
                        for i, choice in enumerate(ordered):
                            d[choice] = i
                        self._ordered_cat_hp[key] = (ordered, d)
                    elif all(
                        isinstance(x, int) or isinstance(x, float)
                        for x in domain.categories
                    ):
                        ordered = sorted(domain.categories)
                        d = {}
                        for i, choice in enumerate(ordered):
                            d[choice] = i
                        self._ordered_choice_hp[key] = (ordered, d)
                    else:
                        self._unordered_cat_hp[key] = len(domain.categories)
                if str(sampler) != "Normal":
                    self._bounded_keys.append(key)
        self._space_keys = sorted(self._tunable_keys)
        if self.prune_attr and self.prune_attr not in self.space and self.max_resource:
            self.min_resource = self.min_resource or self._min_resource()
            self._resource = self._round(self.min_resource)
            self._space_keys.append(self.prune_attr)
        else:
            self._resource = None
        self.incumbent = {}
        self.incumbent = self.normalize(self.best_config)  # flattened
        self.best_obj = self.cost_incumbent = None
        self.dim = len(self._tunable_keys)  # total # tunable dimensions
        self._direction_tried = None
        self._num_complete4incumbent = self._cost_complete4incumbent = 0
        self._num_allowed4incumbent = 2 * self.dim
        self._proposed_by = {}  # trial_id: int -> incumbent: Dict
        self.step_ub = np.sqrt(self.dim)
        self.step = self.STEPSIZE * self.step_ub
        lb = self.step_lower_bound
        if lb > self.step:
            self.step = lb * 2
        # upper bound
        if self.step > self.step_ub:
            self.step = self.step_ub
        # maximal # consecutive no improvements
        self.dir = 2 ** (min(9, self.dim))
        self._configs = {}  # dict from trial_id to (config, stepsize)
        self._K = 0
        self._iter_best_config = 1
        self.trial_count_proposed = self.trial_count_complete = 1
        self._num_proposedby_incumbent = 0
        self._reset_times = 0
        # record intermediate trial cost
        self._trial_cost = {}
        self._same = False  # whether the proposed config is the same as best_config
        self._init_phase = True  # initial phase to increase initial stepsize
        self._trunc = 0
        # no truncation by default. when > 0, it means how many
        # non-zero dimensions to keep in the random unit vector

        self.dir = 2 ** (min(5, self.dim))
        # self.dir = (
        # max(self.dim // 4, 2)  # max number of trials without improvement
        # )

    # copied from FLAML==0.8.0
    @property
    def step_lower_bound(self) -> float:
        step_lb = self._step_lb
        for key in self._tunable_keys:
            if key not in self.best_config:
                continue
            domain = self.space[key]
            sampler = domain.get_sampler()
            # the stepsize lower bound for log uniform variables depends on the
            # current config
            if isinstance(sampler, sample.Quantized):
                q = sampler.q
                sampler_inner = sampler.get_sampler()
                if str(sampler_inner) == "LogUniform":
                    step_lb = min(
                        step_lb,
                        np.log(1.0 + q / self.best_config[key])
                        / np.log(domain.upper / domain.lower),
                    )
            elif isinstance(domain, sample.Integer) and str(sampler) == "LogUniform":
                step_lb = min(
                    step_lb,
                    np.log(1.0 + 1.0 / self.best_config[key])
                    / np.log((domain.upper - 1) / domain.lower),
                )
        if np.isinf(step_lb):
            step_lb = self.STEP_LOWER_BOUND
        else:
            step_lb *= self.step_ub
        return step_lb

    def config_signature(self, config) -> tuple:
        """return the signature tuple of a config"""
        config = flatten_dict(config)
        value_list = []
        for key in self.signature_space:
            if key in config:
                value = config[key]
                if key == self.prune_attr:
                    value_list.append(value)
                # else key must be in self.space
                # get rid of list type or constant,
                # e.g., "eval_metric": ["logloss", "error"]
                elif callable(getattr(self.space[key], "sample", None)):
                    if isinstance(self.space[key], sample.Integer):
                        value_list.append(int(round(value)))
                    else:
                        value_list.append(value)
            else:
                value_list.append(None)
        return tuple(value_list)

    def reach(self, other: Searcher) -> bool:
        """whether the incumbent can reach the incumbent of other"""
        config1, config2 = self.best_config, other.best_config
        incumbent1, incumbent2 = self.incumbent, other.incumbent
        if self.prune_attr not in config1:
            config1[self.prune_attr] = self.min_resource
        if self.prune_attr not in config2:
            config2[self.prune_attr] = other.min_resource
        if self._resource and config1[self.prune_attr] > config2[self.prune_attr]:
            # resource will not decrease
            return False
        if set(self.space) != set(other.space):
            return False
        for key in self._unordered_cat_hp:
            # unordered cat choice is hard to reach by chance
            if config1[key] != config2[key]:
                return False
        delta = np.array(
            [incumbent1[key] - incumbent2[key] for key in self._tunable_keys]
        )
        return np.linalg.norm(delta) <= self.step

    def create(
        self,
        init_config: Dict,
        obj: float,
        cost: float,
        conditional_space=None,
        limit_space_to_init_config: bool = False,
        seed=None,
    ) -> Searcher:
        flow2 = self.__class__(
            init_config,
            self.metric,
            self.mode,
            self._cat_hp_cost,
            self.space,
            self.prune_attr,
            self.min_resource,
            self.max_resource,
            self.resource_multiple_factor,
            seed or self._seed + 1,
            conditional_space=conditional_space,
            limit_space_to_init_config=limit_space_to_init_config,
            cost_attr=self.cost_attr,
            tol=self._tol,
            hyperparameter_names=self._hyperparameter_names,
        )
        flow2.best_obj = obj * self.metric_op  # minimize internally
        flow2.cost_incumbent = cost
        return flow2

    # same as in flaml==0.5.6 with one difference
    def denormalize(self, config):
        """denormalize each dimension in config from [0,1]"""
        config_denorm = {}
        for key, value in config.items():
            if key in self.space:
                # domain: sample.Categorical/Integer/Float/Function
                domain = self.space[key]
                if not callable(getattr(domain, "get_sampler", None)):
                    config_denorm[key] = value
                else:
                    if isinstance(domain, sample.Categorical):
                        # denormalize categorical
                        if key in self._ordered_cat_hp:
                            l, _ = self._ordered_cat_hp[key]
                            n = len(l)
                            config_denorm[key] = l[min(n - 1, int(np.floor(value * n)))]
                        elif key in self._ordered_choice_hp:
                            l, _ = self._ordered_choice_hp[key]
                            n = len(l)
                            config_denorm[key] = l[min(n - 1, int(np.floor(value * n)))]
                        else:
                            assert key in self.incumbent
                            n = self._unordered_cat_hp[key]
                            # copied from flaml==1.0.1
                            if min(n - 1, np.floor(value * n)) == min(
                                n - 1, np.floor(self.incumbent[key] * n)
                            ):
                                config_denorm[key] = self.best_config[key]
                            else:  # ****random value each time!****
                                config_denorm[key] = self._random.choice(
                                    [
                                        x
                                        for x in domain.categories
                                        if x != self.best_config[key]
                                    ]
                                )
                        continue
                    # Uniform/LogUniform/Normal/Base
                    sampler = domain.get_sampler()
                    if isinstance(sampler, sample.Quantized):
                        # sampler is sample.Quantized
                        sampler = sampler.get_sampler()
                    # Handle Log/Uniform
                    if str(sampler) == "LogUniform":
                        config_denorm[key] = (
                            domain.upper / domain.lower
                        ) ** value * domain.lower
                    elif str(sampler) == "Uniform":
                        config_denorm[key] = (
                            value * (domain.upper - domain.lower) + domain.lower
                        )
                    elif str(sampler) == "Normal":
                        # denormalization for 'Normal'
                        config_denorm[key] = value * sampler.sd + sampler.mean
                    else:
                        config_denorm[key] = value
                    # Handle quantized
                    sampler = domain.get_sampler()
                    if isinstance(sampler, sample.Quantized):
                        config_denorm[key] = (
                            np.round(np.divide(config_denorm[key], sampler.q))
                            * sampler.q
                        )
                    # Handle int (4.6 -> 5)
                    if isinstance(domain, sample.Integer):
                        config_denorm[key] = int(round(config_denorm[key]))
            else:  # prune_attr
                config_denorm[key] = value
        return config_denorm

    def _round(self, resource) -> float:
        """round the resource to self.max_resource if close to it"""
        return np.around(super()._round(resource), 2)

    def _is_trial_better(self, trial_id: str, obj: float) -> bool:
        if (
            self._resource
            and self._resource < self._configs[trial_id][0][self.prune_attr]
        ):
            diff = 0
        else:
            diff = np.abs(self.best_obj * self._tol)
        return self.best_obj is None or obj < (self.best_obj - diff)

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        """compare with incumbent"""
        # if better, move, reset num_complete and num_proposed
        # if not better and num_complete >= 2*dim, num_allowed += 2
        self.trial_count_complete += 1
        if not error and result:
            obj = result.get(self._metric)
            if obj:
                obj *= self.metric_op
                if self._is_trial_better(trial_id, obj):
                    self.best_obj = obj
                    self.best_config, self.step = self._configs[trial_id]
                    self.incumbent = self.normalize(self.best_config)
                    self.cost_incumbent = result.get(self.cost_attr)
                    if self._resource:
                        self._resource = self.best_config[self.prune_attr]
                    self._num_complete4incumbent = 0
                    self._cost_complete4incumbent = 0
                    self._num_proposedby_incumbent = 0
                    self._num_allowed4incumbent = 2 * self.dim
                    self._proposed_by.clear()
                    if self._K > 0:
                        # self._oldK must have been set when self._K>0
                        self.step *= np.sqrt(self._K / self._oldK)
                    if self.step > self.step_ub:
                        self.step = self.step_ub
                    self._iter_best_config = self.trial_count_complete
                    if self._trunc:
                        self._trunc = min(self._trunc + 1, self.dim)
                    return
                elif self._trunc:
                    self._trunc = max(self._trunc >> 1, 1)
        proposed_by = self._proposed_by.get(trial_id)
        if proposed_by == self.incumbent:
            # proposed by current incumbent and no better
            self._num_complete4incumbent += 1
            cost = (
                result.get(self.cost_attr) if result else self._trial_cost.get(trial_id)
            )
            if cost:
                self._cost_complete4incumbent += cost
            if (
                self._num_complete4incumbent >= 2 * self.dim
                and self._num_allowed4incumbent == 0
            ):
                self._num_allowed4incumbent = 2
            if self._num_complete4incumbent == self.dir and (
                not self._resource or self._resource == self.max_resource
            ):
                # check stuck condition if using max resource
                self._num_complete4incumbent -= 2
                if self._num_allowed4incumbent < 2:
                    self._num_allowed4incumbent = 2
        # elif proposed_by: del self._proposed_by[trial_id]

    # def suggest(self, trial_id: str) -> Optional[Dict]:
    #     """suggest a new config, one of the following cases:
    #     1. same incumbent, increase resource
    #     2. same resource, move from the incumbent to a random direction
    #     3. same resource, move from the incumbent to the opposite direction
    #     """
    #     if (
    #         self.saved_resource is None
    #         and self._num_complete4incumbent > 0
    #         and self.cost_incumbent
    #         and self._resource
    #         and self._resource < self.max_resource
    #         and (
    #             self._cost_complete4incumbent
    #             >= self.cost_incumbent * self.resource_multiple_factor
    #         )
    #     ):
    #         # consider increasing resource using sum eval cost of complete
    #         # configs
    #         self._resource = self._round(self._resource * self.resource_multiple_factor)
    #         config = self.best_config.copy()
    #         config[self.prune_attr] = self._resource
    #         print(f"{trial_id} increasing resource to {self._resource}")
    #         self.saved_resource = self._resource
    #         # self.incumbent[self.prune_attr] = self._resource
    #         self._direction_tried = None
    #         self._configs[trial_id] = config
    #         return config
    #     self._num_allowed4incumbent -= 1
    #     move = self.incumbent.copy()
    #     if self._direction_tried is not None:
    #         # return negative direction
    #         for i, key in enumerate(self._tunable_keys):
    #             move[key] -= self._direction_tried[i]
    #         self._direction_tried = None
    #     else:
    #         # propose a new direction
    #         self._direction_tried = self.rand_vector_unit_sphere(self.dim) * self.step
    #         for i, key in enumerate(self._tunable_keys):
    #             move[key] += self._direction_tried[i]
    #     self._project(move)
    #     config = self.denormalize(move)
    #     self._proposed_by[trial_id] = self.incumbent
    #     self._configs[trial_id] = config
    #     return unflatten_dict(config)

    # def on_trial_complete(
    #     self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    # ):
    #     """compare with incumbent"""
    #     # if better, move, reset num_complete and num_proposed
    #     # if not better and num_complete >= 2*dim, num_allowed += 2
    #     try:
    #         self.trial_count += 1
    #     except AttributeError:
    #         self.trial_count = 1
    #     print("flow2 on_trial_complete")
    #     if (
    #         self.saved_resource is not None
    #         and result.get(self.prune_attr) >= self.saved_resource - 1e-10
    #     ):
    #         print(
    #             f"{trial_id} reseting saved resource due to {result.get(self.prune_attr)}"
    #         )
    #         self.saved_resource = None
    #     if not error and result:
    #         obj = result.get(self._metric)
    #         if obj:
    #             obj *= self.metric_op
    #             if self.best_obj is None or obj < self.best_obj - (
    #                 0
    #                 if not self._resource
    #                 or self._resource < self._configs[trial_id][self.prune_attr]
    #                 else np.abs(self.best_obj * self._tol)
    #             ):
    #                 self.best_obj, self.best_config = obj, self._configs[trial_id]
    #                 self.incumbent = self.normalize(self.best_config)
    #                 self.cost_incumbent = result.get(self.cost_attr)
    #                 if self._resource:
    #                     self._resource = self.best_config[self.prune_attr]
    #                 self._num_complete4incumbent = 0
    #                 self._cost_complete4incumbent = 0
    #                 self._num_allowed4incumbent = 2 * self.dim
    #                 self._proposed_by.clear()
    #                 if self._K > 0:
    #                     self.step *= np.sqrt(self._K / self._oldK)
    #                 if self.step > self.step_ub:
    #                     self.step = self.step_ub
    #                 self._iter_best_config = self.trial_count
    #                 return
    #     proposed_by = self._proposed_by.get(trial_id)
    #     if proposed_by == self.incumbent:
    #         # proposed by current incumbent and no better
    #         print("proposed by current incumbent and no better")
    #         self._num_complete4incumbent += 1
    #         cost = (
    #             result.get(self.cost_attr) if result else self._trial_cost.get(trial_id)
    #         )
    #         if cost:
    #             self._cost_complete4incumbent += cost
    #         print(
    #             f"cost={cost}, cost_incumbent={self.cost_incumbent}, _cost_complete4incumbent={self._cost_complete4incumbent}"
    #         )
    #         print(
    #             f"num_complete4incumbent={self._num_complete4incumbent}, self.cost_incumbent * self.resource_multiple_factor={self.cost_incumbent * self.resource_multiple_factor}"
    #         )
    #         if (
    #             self._num_complete4incumbent >= 2 * self.dim
    #             and self._num_allowed4incumbent == 0
    #         ):
    #             self._num_allowed4incumbent = 2
    #         if self._num_complete4incumbent >= self.dir and (
    #             not self._resource or self._resource >= self.max_resource
    #         ):
    #             print(
    #                 f"step={self.step}, lb={self.step_lower_bound} _oldK={self._iter_best_config} _K={self.trial_count + 1}"
    #             )
    #             # check stuck condition if using max resource
    #             if self.step >= self.step_lower_bound:
    #                 # decrease step size
    #                 self._oldK = self._K if self._K else self._iter_best_config
    #                 self._K = self.trial_count + 1
    #                 self.step *= np.sqrt(self._oldK / self._K)
    #             self._num_complete4incumbent -= 2
    #             if self._num_allowed4incumbent < 2:
    #                 self._num_allowed4incumbent = 2
    #     # elif proposed_by: # proposed by older incumbent
    #     #     del self._proposed_by[trial_id]


LocalSearch = PatchedFLOW2


class SharingSearchThread(SearchThread):
    """Class of global or local search thread"""

    def __init__(
        self,
        mode: str = "min",
        search_alg: Optional[Searcher] = None,
        cost_attr: Optional[str] = "time_total_s",
        prune_attr=None,
        max_prune_attr=None,
    ):
        """When search_alg is omitted, use local search FLOW2"""
        self._search_alg = search_alg
        self._is_ls = isinstance(search_alg, FLOW2)
        self._mode = mode
        self._metric_op = 1 if mode == "min" else -1
        self.cost_best = self.cost_last = self.cost_total = self.cost_best1 = getattr(
            search_alg, "cost_incumbent", 0
        )
        self.cost_best2 = 0
        self.obj_best1 = self.obj_best2 = getattr(
            search_alg, "best_obj", np.inf
        )  # inherently minimize
        # eci: expected cost for improvement
        self.eci = self.cost_best
        self.priority = self.speed = 0
        self._running = set()
        self._init_config = True
        if cost_attr:
            self.cost_attr = cost_attr
        self.prune_attr = prune_attr
        if prune_attr:
            assert max_prune_attr
        self.max_prune_attr = max_prune_attr
        self.last_prune_attr = 1.0
        self.resources = [1.0]

    @property
    def running(self) -> int:
        return len(self._running) or 1

    @property
    def estimator(self) -> str:
        if self._is_ls:
            return self._search_alg.space["Estimator"]
        return None

    def __repr__(self) -> str:
        if self._is_ls:
            return (
                f"SharingSearchThread with FLOW2 {self._search_alg.space['Estimator']}"
            )
        return f"SharingSearchThread with {self._search_alg}"

    def on_trial_complete(
        self,
        trial_id: str,
        result: Optional[Dict] = None,
        error: bool = False,
        thread_created: bool = True,
        update_results: bool = True,
        add_to_gs: bool = True,
    ) -> bool:
        """update the statistics of the thread"""
        print(f"on_trial_complete {trial_id} {self._search_alg}")
        if not self._search_alg:
            return False
        if pd.isnull(result.get(self._search_alg.metric, None)):
            error = True
        print(
            f"on_trial_complete error: {error} thread_created: {thread_created} update_results: {update_results}"
        )
        if not hasattr(self._search_alg, "_ot_trials"):
            # optuna doesn't handle error
            if self._is_ls or not self._init_config:
                self._search_alg.on_trial_complete(
                    trial_id,
                    result,
                    error,
                )
            else:
                # init config is not proposed by self._search_alg
                # under this thread
                self._init_config = False
        elif not error:
            if self.prune_attr in result:
                resource = result[self.prune_attr]
                if resource not in self.resources:
                    self.resources.append(resource)
                    self.resources.sort()
                max_prune_attr_reached = resource >= self.max_prune_attr
            else:
                resource = 1.0
                max_prune_attr_reached = True
            print(
                f"resource: {resource} max_prune_attr_reached: {max_prune_attr_reached}"
            )
            if self.prune_attr in result:
                result = result.copy()
                result.pop(self.prune_attr)
            try:
                if trial_id in self._search_alg._ot_trials:
                    print("adding trial result to optuna")
                    self._search_alg.on_trial_complete(
                        trial_id,
                        result,
                        state=TrialState.COMPLETE
                        if max_prune_attr_reached
                        else TrialState.PRUNED,
                        num_intermediate_values=self.resources.index(resource) + 1,
                    )
                    print(
                        f"Optuna has {len(self._search_alg._ot_study.trials)} ({len(self._search_alg._ot_study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))} usable) trials in memory"
                    )
            except Exception as e:
                logger.debug(
                    f"couldn't add trial {result} to optuna.\n{traceback.format_exc()}"
                )
                print(
                    f"couldn't add trial {result} to optuna.\n{traceback.format_exc()}"
                )

        if update_results and result and not error:
            print(
                f"updating results {trial_id} {self._search_alg}, searcher metric {self._search_alg.metric} {result[self._search_alg.metric]}"
            )
            if self.prune_attr in result:
                self.last_prune_attr = result[self.prune_attr]
            self.cost_last = result.get(self.cost_attr, 1)
            self.cost_total += self.cost_last
            print(
                f"{str(self)} cost_last={self.cost_last} cost_total={self.cost_total}"
            )
            # if not isinstance(self._search_alg, FLOW2):
            #     logger.info(f"result.metric{result[self._search_alg.metric]}")
            if self._search_alg.metric in result:
                obj = result[self._search_alg.metric] * self._metric_op
                if obj < self.obj_best1:
                    self.cost_best2 = self.cost_best1
                    self.cost_best1 = self.cost_total
                    self.obj_best2 = obj if np.isinf(self.obj_best1) else self.obj_best1
                    self.obj_best1 = obj
                    self.cost_best = self.cost_last
            self._update_speed()
            ret = True
        else:
            ret = False
        try:
            self._running.remove(trial_id)
        except KeyError:
            pass
        return ret

    def _update_speed(self):
        old_speed = self.speed
        assert self.running > 0
        super()._update_speed()
        print(f"{str(self)} old speed={old_speed} new speed={self.speed}")

    def suggest(self, trial_id: str, **kwargs) -> Optional[Dict]:
        """use the suggest() of the underlying search algorithm"""
        if isinstance(self._search_alg, FLOW2):
            config = self._search_alg.suggest(trial_id, **kwargs)
        else:
            # config = self._search_alg.suggest(trial_id)
            try:
                config = self._search_alg.suggest(trial_id, **kwargs)
            except:
                traceback.print_exc()
                logger.warning(
                    "The global search method raises error. "
                    "Ignoring for this iteration.\n"
                    f"{traceback.format_exc()}"
                )
                config = None
        if config is not None:
            self._running.add(trial_id)
        return config

    def on_trial_result(self, trial_id: str, result: Dict):
        """TODO update the statistics of the thread with partial result?"""
        # logger.debug('[SearchThread] on trial result')
        if not self._search_alg:
            return

        if not hasattr(self._search_alg, "_ot_trials"):
            # optuna doesn't handle error
            self._search_alg.on_trial_result(
                trial_id,
                result,
            )
        elif trial_id in self._search_alg._ot_trials:
            self._search_alg.on_trial_result(
                trial_id,
                result,
            )
        if self.cost_attr in result and self.cost_last < result[self.cost_attr]:
            self.cost_last = result[self.cost_attr]
            # self._update_speed()


class EstimatorState(Bunch):
    def __init__(self, resource):
        super().__init__(
            resource=resource,
            best_score=np.inf,
            best_score_old=np.inf,
            best_cost=0,
            best_cost_old=0,
            best_eval_time=0,
            total_time=0,
        )

    @property
    def eci(self):
        return max(
            self.best_cost - self.best_cost_old, self.total_time - self.best_cost
        )


class ConditionalBlendSearch(BlendSearch):
    """class for BlendSearch algorithm"""

    _FORCE_GS_EVERY_N_ITER = 64
    _MAX_GS_RETRIES = 3

    def __init__(
        self,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        space: Optional[dict] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        secondary_points_to_evaluate: Optional[List[Dict]] = None,
        low_cost_partial_config: Optional[dict] = None,
        prune_attr: Optional[str] = None,
        min_resource: Optional[float] = None,
        max_resource: Optional[float] = None,
        reduction_factor: Optional[float] = None,
        resources_per_trial: Optional[dict] = None,
        global_search_alg: Optional[Searcher] = None,
        config_constraints: Optional[
            List[Tuple[Callable[[dict], float], str, float]]
        ] = None,
        metric_constraints: Optional[List[Tuple[str, str, float]]] = None,
        time_attr: str = "time_total_s",
        seed: Optional[int] = None,
        use_extended: bool = False,
        spawn_local_threads: bool = True,
        mem_size=None,
        meta_info: Optional[dict] = None,
    ):
        self._metric, self._mode = metric, mode
        self._config_constraints = config_constraints
        self._metric_constraints = metric_constraints
        self._conditional_space = get_conditions(
            space, to_str=True, use_extended=use_extended
        )
        self._time_attr = time_attr
        self._meta_info = meta_info or {}

        self._seed = seed

        init_config = self._get_all_default_values(
            space,
            get_categorical=False,
            use_extended=use_extended,
            only_cost_related=True,
        )
        cost_bounds = self._get_cost_related_bounds(
            space,
            use_extended=use_extended,
        )
        self._cost_bounds = {k: v for k, v in cost_bounds.items() if k in init_config}
        tunable_space, _, hyperparameter_names = get_all_tunable_params(
            space, to_str=True, use_extended=use_extended
        )
        tune_space = get_tune_distributions(tunable_space)

        if global_search_alg is not None:
            self._gs = global_search_alg
        elif getattr(self, "__name__", None) != "CFO":
            self._gs = GlobalSearch(
                space=space,
                metric=metric,
                mode=mode,
                seed=seed,
                n_startup_trials=10,
                use_extended=use_extended,
                remove_const_values=True,
            )
        else:
            self._gs = None

        const_values = {
            k
            for k, v in tune_space.items()
            if isinstance(v, Categorical) and len(v.categories) == 1
        }
        tune_space = {k: v for k, v in tune_space.items() if k not in const_values}
        if points_to_evaluate:
            points_to_evaluate = [
                {k: v for k, v in point.items() if k in tune_space}
                for point in points_to_evaluate
            ]

        if secondary_points_to_evaluate:
            secondary_points_to_evaluate = [
                {k: v for k, v in point.items() if k in tune_space}
                for point in secondary_points_to_evaluate
            ]

        self._points_to_evaluate = points_to_evaluate
        self._points_to_evaluate_len = len(points_to_evaluate)
        self._secondary_points_to_evaluate = secondary_points_to_evaluate
        self._init_finished = False
        self._points_to_evaluate_trials = {}
        self._ls = LocalSearch(
            init_config=init_config,
            metric=metric,
            mode=mode,
            space=tune_space,
            prune_attr=prune_attr,
            min_resource=min_resource,
            max_resource=max_resource,
            resource_multiple_factor=reduction_factor,
            seed=seed,
            cost_attr=self._time_attr,
            hyperparameter_names=hyperparameter_names,
        )
        self._resources_per_trial = resources_per_trial
        self._mem_size = mem_size
        self._mem_threshold = (
            resources_per_trial.get("mem") if resources_per_trial else None
        )
        self._reached_max_prune_attr = not bool(prune_attr)
        self._random = np.random.RandomState(seed)
        self._estimator_states = {
            k: EstimatorState(resource=min_resource if prune_attr else max_resource)
            for k in tunable_space["Estimator"].values
        }
        self._spawn_local_threads = spawn_local_threads
        self._init_search()

    @property
    def _conditional_space_estimators(self):
        return {"Estimator": self._conditional_space["Estimator"]}

    def _get_cost_related_bounds(
        self,
        pipeline_blueprint,
        use_extended=False,
    ) -> dict:
        default_grid = {
            k: v
            for k, v in pipeline_blueprint.get_all_distributions(
                use_extended=use_extended
            ).items()
        }
        default_values = {}
        for k, v in default_grid.items():
            for v2 in v.values:
                for k3, v3 in v2.get_tuning_grid(use_extended=use_extended).items():
                    if not v3.cost_related:
                        continue
                    name = v2.get_hyperparameter_key_suffix(k, k3)
                    default_values[name] = v3.cost_bounds
        return default_values

    def _get_all_default_values(
        self,
        pipeline_blueprint,
        get_categorical=True,
        use_extended=False,
        only_cost_related=False,
    ) -> dict:
        default_grid = {
            k: v
            for k, v in pipeline_blueprint.get_all_distributions(
                use_extended=use_extended
            ).items()
        }
        default_values = {}
        for k, v in default_grid.items():
            for v2 in v.values:
                for k3, v3 in v2.get_tuning_grid(use_extended=use_extended).items():
                    if not get_categorical and isinstance(v3, CategoricalDistribution):
                        continue
                    if only_cost_related and not v3.cost_related:
                        continue
                    name = v2.get_hyperparameter_key_suffix(k, k3)
                    default_values[name] = v3.default
        return default_values

    @property
    def keys_to_keep(self):
        keys = set(self._ls.space)
        if self._ls.prune_attr:
            keys.add(self._ls.prune_attr)
        return keys

    def _init_search(self):
        """initialize the search"""
        super()._init_search()
        self._suggested_configs = {}
        self._search_thread_pool = {
            # id: int -> thread: SearchThread
            0: SharingSearchThread(
                self._ls.mode,
                self._gs,
                cost_attr=self._time_attr,
                prune_attr=self._ls.prune_attr,
                max_prune_attr=self._ls.max_resource,
            )
        }
        self._iters_without_new_best = 0

    @property
    def _metric_target_sign(self):
        return self._metric_target * self._ls.metric_op

    @property
    def _global_search_thread(self):
        return self._search_thread_pool.get(0, None)

    def save(self, checkpoint_path: str):
        save_object = (
            self._metric_target,
            self._search_thread_pool,
            self._thread_count,
            self._init_used,
            self._trial_proposed_by,
            self._ls_bound_min,
            self._ls_bound_max,
            self._result,
            self._deadline,
            self._suggested_configs,
            self._conditional_space,
            self._time_attr,
            self._iters_without_new_best,
            self._points_to_evaluate,
            self._points_to_evaluate_len,
            self._secondary_points_to_evaluate,
            self._init_finished,
            self._points_to_evaluate_trials,
            self._reached_max_prune_attr,
            self._cost_bounds,
            self._random,
            self._spawn_local_threads,
            self._meta_info
            # self._diversification_multipliers,
        )
        with open(checkpoint_path, "wb") as outputFile:
            cloudpickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = cloudpickle.load(inputFile)
        (
            self._metric_target,
            self._search_thread_pool,
            self._thread_count,
            self._init_used,
            self._trial_proposed_by,
            self._ls_bound_min,
            self._ls_bound_max,
            self._result,
            self._deadline,
            self._suggested_configs,
            self._conditional_space,
            self._time_attr,
            self._iters_without_new_best,
            self._points_to_evaluate,
            self._points_to_evaluate_len,
            self._secondary_points_to_evaluate,
            self._init_finished,
            self._points_to_evaluate_trials,
            self._reached_max_prune_attr,
            self._cost_bounds,
            self._random,
            self._spawn_local_threads,
            self._meta_info
            # self._diversification_multipliers,
        ) = save_object

    def restore_from_dir(self, checkpoint_dir: str):
        super.restore_from_dir(checkpoint_dir)

    def on_trial_result(self, trial_id: str, result: Dict):
        if trial_id not in self._trial_proposed_by:
            return
        thread_id = self._trial_proposed_by[trial_id]
        if thread_id not in self._search_thread_pool:
            return
        config = {
            f"config/{k}": v for k, v in self._suggested_configs[trial_id].items()
        }
        result = {**config, **result}
        self._search_thread_pool[thread_id].on_trial_result(trial_id, result)

    def on_trial_complete(
        self,
        trial_id: str,
        result: Optional[dict] = None,
        error: bool = False,
        condition_kwargs=None,
    ):
        if result is None:
            result = {}

        finished_count = None
        if not self._init_finished:
            self._points_to_evaluate_trials[trial_id] = (
                trial_id,
                result,
                error,
                condition_kwargs,
            )
            finished_count = sum(
                1
                for i in self._points_to_evaluate_trials.values()
                if i[1].get(f"config/{META_KEY}/init", True)
            )
            self._init_finished = finished_count >= self._points_to_evaluate_len

        print(
            f"self._init_finished {self._init_finished} finished_count {finished_count}  self._points_to_evaluate_len { self._points_to_evaluate_len}"
        )

        if self._init_finished:
            # TODO if we are caching and without early stopping, consider only estimator fit time for priority
            # TODO when multiple pipelines have the same estimator, leave only the best (delete the threads)
            if self._points_to_evaluate_trials:
                clean_sorted_evaluted_trials = sorted(
                    [
                        trial
                        for trial in self._points_to_evaluate_trials.values()
                        if not pd.isnull(trial[1].get(self._metric, None))
                    ],
                    key=lambda trial: trial[1][self._metric] * self._ls.metric_op,
                    reverse=True,
                )

                median_metric = np.percentile(
                    [
                        trial[1][self._metric] * self._ls.metric_op
                        for trial in clean_sorted_evaluted_trials
                    ],
                    75,
                )

                cutoff_trial = next(
                    trial
                    for trial in clean_sorted_evaluted_trials
                    if (trial[1][self._metric] * self._ls.metric_op) <= median_metric
                )
                # cutoff_trial = clean_sorted_evaluted_trials[
                #    len(clean_sorted_evaluted_trials) // 2
                # ]
                self._points_to_evaluate_trials.pop(cutoff_trial[0])
                self._on_trial_complete(
                    trial_id=cutoff_trial[0],
                    result=cutoff_trial[1],
                    error=cutoff_trial[2],
                    condition_kwargs=cutoff_trial[3],
                )
                for trial in clean_sorted_evaluted_trials:
                    if trial[0] == cutoff_trial[0]:
                        continue
                    self._on_trial_complete(
                        trial_id=trial[0],
                        result=trial[1],
                        error=trial[2],
                        condition_kwargs=trial[3],
                    )
                self._points_to_evaluate_trials = None
                print(self._secondary_points_to_evaluate)
                self._secondary_points_to_evaluate = [
                    point
                    for point in self._secondary_points_to_evaluate
                    if point["Estimator"]
                    in [
                        thread.estimator
                        for thread_id, thread in self._search_thread_pool.items()
                    ]
                ]
                self._points_to_evaluate = self._secondary_points_to_evaluate
                self._iters_without_new_best = 0
                # self._last_global_search = np.inf
                # _, _, local_threads_by_priority = self._select_thread()
                # for thread_id, _, _ in local_threads_by_priority[2:]:
                #    del self._search_thread_pool[thread_id]
            else:
                return self._on_trial_complete(
                    trial_id=trial_id,
                    result=result,
                    error=error,
                    condition_kwargs=condition_kwargs,
                )

    def _on_trial_complete(
        self,
        trial_id: str,
        result: Optional[dict] = None,
        error: bool = False,
        condition_kwargs=None,
    ):
        if result is None:
            result = {}

        thread_id = self._trial_proposed_by[trial_id]
        condition_kwargs = condition_kwargs or {}
        create_condition = (
            result
            and not thread_id
            and self._create_condition(result, **condition_kwargs)
        )
        print(
            f"on_trial_complete thread_id {thread_id}, self._search_thread_pool {self._search_thread_pool}"
        )
        if thread_id in self._search_thread_pool:
            thread = self._search_thread_pool[thread_id]
            was_updated = thread.on_trial_complete(
                trial_id, result, error, thread_created=create_condition
            )
            treat_as_thread_created = self._init_finished and create_condition
            if was_updated:
                estimator_state = self._estimator_states[
                    self._suggested_configs[trial_id]["Estimator"]
                ]
                obj = result[self._metric] * self._ls.metric_op
                estimator_state.total_time += thread.cost_last
                if obj < estimator_state.best_score:
                    estimator_state.best_score_old = (
                        estimator_state.best_score
                        if estimator_state.best_score < np.inf
                        else 2 * obj
                    )
                    estimator_state.best_score = obj
                    estimator_state.best_cost_old = estimator_state.best_cost
                    estimator_state.best_cost = estimator_state.total_time
                    estimator_state.best_eval_time = thread.cost_last
                if (
                    estimator_state.resource is not None
                    and estimator_state.resource < thread.last_prune_attr
                ):
                    estimator_state.resource = thread.last_prune_attr
            if thread_id > 0 and self._global_search_thread:
                self._global_search_thread.on_trial_complete(
                    trial_id,
                    result,
                    error,
                    thread_created=treat_as_thread_created,
                    update_results=False,
                )
            del self._trial_proposed_by[trial_id]
            # if not thread_id: logger.info(f"result {result}")
        if result:
            config = self._suggested_configs[trial_id]
            (
                _,
                config_signature,
            ) = self._has_config_been_already_tried(config)
            if error:  # remove from result cache
                del self._result[config_signature]
            else:  # add to result cache
                self._result[config_signature] = result
            # update target metric if improved
            if (result[self._metric] - self._metric_target) * self._ls.metric_op < 0:
                self._metric_target = result[self._metric]
                self._iters_without_new_best = 0
            if create_condition:
                # thread creator
                self._search_thread_pool[self._thread_count] = SharingSearchThread(
                    self._ls.mode,
                    self._ls.create(
                        config,
                        result[self._metric],
                        cost=result[self._time_attr],
                        conditional_space=self._conditional_space,
                        limit_space_to_init_config=True,
                        seed=self._ls._seed + self._thread_count,
                    ),
                    cost_attr=self._time_attr,
                    prune_attr=self._ls.prune_attr,
                    max_prune_attr=self._ls.max_resource,
                )
                self._search_thread_pool[
                    self._thread_count
                ]._search_alg.cost_attr = self._time_attr
                thread_id = self._thread_count
                self._thread_count += 1

                self._update_admissible_region(
                    config, self._ls_bound_min, self._ls_bound_max
                )
                print("\nGS BOUNDS UPDATE")
                print(self._ls.denormalize(self._ls_bound_min))
                print(self._ls.denormalize(self._ls_bound_max))
                print("")
            # reset admissible region to ls bounding box
            self._gs_admissible_min.update(self._ls_bound_min)
            self._gs_admissible_max.update(self._ls_bound_max)

        # cleaner
        # logger.info(f"thread {thread_id} in search thread pool="
        #     f"{thread_id in self._search_thread_pool}")

        # TODO add cleaned trials to Optuna
        if thread_id and thread_id in self._search_thread_pool:
            # local search thread
            self._clean(thread_id)

    def _create_condition(self, result: dict, median=None) -> bool:
        """create thread condition"""
        if not self._spawn_local_threads:
            return False
        if len(self._search_thread_pool) < 2:
            return True
        obj_median = median or np.median(
            [thread.obj_best1 for id, thread in self._search_thread_pool.items() if id]
        )
        print(
            f"create condition: {result[self._metric] * self._ls.metric_op} must be lower than {obj_median}"
        )
        return result[self._metric] * self._ls.metric_op < obj_median

    def _update_admissible_region(self, config, admissible_min, admissible_max):
        # update admissible region
        print(f"Updating admissible region: {config}")
        normalized_config = self._ls.normalize(config)
        for key in admissible_min:
            if key in config:
                value = normalized_config[key]
                if value > admissible_max[key]:
                    admissible_max[key] = value
                elif value < admissible_min[key]:
                    admissible_min[key] = value

    def _valid(self, config: dict, step_multiplier=1) -> bool:
        """config validator"""
        step_size = self._ls.STEPSIZE * step_multiplier
        admissible_min_denorm = self._ls.denormalize(
            {k: max(0, v - step_size) for k, v in self._gs_admissible_min.items()}
        )
        admissible_max_denorm = self._ls.denormalize(
            {k: min(1, v + step_size) for k, v in self._gs_admissible_max.items()}
        )
        for key in self._gs_admissible_min:
            if key in config:
                value = config[key]
                # logger.info(
                #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                if (
                    self._cost_bounds[key] != "upper"
                    and value < admissible_min_denorm[key]
                ):
                    print(
                        f"key {key} is invalid due to {value} < {admissible_min_denorm[key]}"
                    )
                    # print(f"suggested config {config}")
                    # print(f"valid config {self._make_config_valid(config)}")
                    return False
                elif (
                    self._cost_bounds[key] != "lower"
                    and value > admissible_max_denorm[key]
                ):
                    print(
                        f"key {key} is invalid due to {value} > {admissible_max_denorm[key]}"
                    )
                    # print(f"suggested config {config}")
                    # print(f"valid config {self._make_config_valid(config)}")
                    return False
        return True

    def _make_config_valid(self, config: dict, step_multiplier=1):
        normalized_config = self._ls.normalize(config)
        step_size = self._ls.STEPSIZE * step_multiplier
        enforced_values = {}
        for key in self._gs_admissible_min:
            if key in config:
                value = normalized_config[key]
                # logger.info(
                #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                if (
                    self._cost_bounds[key] != "upper"
                    and value + step_size < self._gs_admissible_min[key]
                ):
                    enforced_values[key] = max(
                        0, self._gs_admissible_min[key] - step_size
                    )
                elif (
                    self._cost_bounds[key] != "lower"
                    and value > self._gs_admissible_max[key] + step_size
                ):
                    enforced_values[key] = min(
                        1, self._gs_admissible_max[key] + step_size
                    )
        print(f"enforced values: {enforced_values}")
        new_config = {**config, **self._ls.denormalize(enforced_values)}
        return new_config

    def _select_estimator(self, local_threads):
        if not self._estimator_states or not local_threads:
            return None, None
        inv = []
        estimators_in_threads = {
            thread_tuple[1].estimator for thread_tuple in local_threads
        }
        print(f"best global score={self._metric_target_sign}")
        for estimator, state in self._estimator_states.items():
            print(
                f"estimator {estimator} (skipped {estimator not in estimators_in_threads})"
            )
            print(
                f"best_score={state.best_score} best_score_old={state.best_score_old}"
            )
            print(f"best_cost={state.best_cost} best_cost_old={state.best_cost_old}")
            print(
                f"total_time={state.total_time} best_eval_time={state.best_eval_time}"
            )
            if estimator not in estimators_in_threads:
                inv.append(0)
                continue
            estimated_cost = state.eci
            if state.resource is not None and state.resource < self._ls.max_resource:
                estimated_cost = min(
                    estimated_cost,
                    state.best_eval_time
                    * min(
                        self._ls.resource_multiple_factor,
                        self._ls.max_resource / state.resource,
                    ),
                )
            gap = state.best_score - (self._metric_target_sign)
            if gap > 0:
                delta_loss = (
                    state.best_score_old - state.best_score
                ) or state.best_score
                delta_time = (state.total_time - state.best_cost_old) or 1e-10
                speed = delta_loss / delta_time
                if speed:
                    estimated_cost = max(2 * gap / speed, estimated_cost)
            if estimated_cost == 0:
                estimated_cost = 1e-10
            inv.append(1 / estimated_cost)
        s = sum(inv)
        inv = [i / s for i in inv]
        p = self._random.rand()
        q = 0
        estimator_list = list(self._estimator_states.keys())
        prios = dict(zip(estimator_list, inv))
        print(f"estimator priorities: {prios}")
        for i in range(len(inv)):
            if inv[i]:
                q += inv[i]
                if p < q:
                    return estimator_list[i], prios
        return None, None

    def _select_thread(self) -> Tuple:
        """thread selector; use can_suggest to check LS availability"""
        # update priority
        min_eci = self._deadline - time.time()
        print(f"DEADLINE: {min_eci}")
        # if min_eci <= 0:
        #    time.sleep(1)
        #    return -1, -1, []
        max_speed = 0
        for thread in self._search_thread_pool.values():
            if thread.speed > max_speed:
                max_speed = thread.speed
        for thread in self._search_thread_pool.values():
            thread.update_eci(self._metric_target, max_speed)
            if thread.eci < min_eci:
                min_eci = thread.eci
        for thread in self._search_thread_pool.values():
            thread.update_priority(min_eci)

        top_thread_id = backup_thread_id = 0
        priority1 = self._search_thread_pool[0].priority
        # print(f"priority of thread 0={priority1}, obj_best1={self._search_thread_pool[0].obj_best1}")
        # diversification_multiplier = 0.99
        # print(f"diversification: {self._diversification_multipliers}")

        local_threads_by_priority = sorted(
            [
                (
                    thread_id,
                    thread,
                    thread.priority,
                    # * (
                    #    diversification_multiplier
                    #    ** self._diversification_multipliers.get(thread_id, 0)
                    # ),
                )
                for thread_id, thread in self._search_thread_pool.items()
                if thread_id and thread.can_suggest
            ],
            reverse=True,
            key=lambda x: x[2],
        )

        estimator_to_use, estimator_priorities = self._select_estimator(
            local_threads_by_priority
        )

        if not estimator_to_use:
            local_threads_by_priority_estimator_only = local_threads_by_priority
        else:
            local_threads_by_priority_estimator_only = [
                thread_tuple
                for thread_tuple in local_threads_by_priority
                if thread_tuple[1].estimator == estimator_to_use
            ]
            if not local_threads_by_priority_estimator_only:
                local_threads_by_priority_estimator_only = local_threads_by_priority

        local_threads_by_priority_estimator_only = sorted(
            local_threads_by_priority_estimator_only,
            reverse=True,
            key=lambda x: x[2],
        )

        local_threads_by_priority_with_estimator = sorted(
            [
                (
                    thread_id,
                    thread,
                    thread_priority,
                    thread_priority * estimator_priorities[thread.estimator],
                )
                for thread_id, thread, thread_priority in local_threads_by_priority
            ],
            reverse=True,
            key=lambda x: x[2],
        )
        print(
            f"global search priority: {priority1}, ls {local_threads_by_priority_with_estimator[0][2]}"
        )
        # print(local_threads_by_priority)
        print(local_threads_by_priority_with_estimator)
        print(local_threads_by_priority_estimator_only)

        if local_threads_by_priority_estimator_only:
            top_thread_id = (
                0
                if priority1 >= local_threads_by_priority_with_estimator[0][2]
                else local_threads_by_priority_estimator_only[0][0]
            )
            backup_thread_id = local_threads_by_priority_estimator_only[0][0]

        return top_thread_id, backup_thread_id, local_threads_by_priority_estimator_only

    def _suggest_from_points_to_evaluate(self, trial_id: str):
        init_config = (
            self._points_to_evaluate.pop(0)
            if self._points_to_evaluate
            else self._ls.init_config
        )
        config = self._ls.complete_config(
            init_config, self._ls_bound_min, self._ls_bound_max
        )
        prune_attr = config.get(self._ls.prune_attr, None) if config else None
        assert len(config) == len(self._ls.space) + int(bool(self._ls.prune_attr))
        # logger.info(f"reset config to {config}")
        self._init_used = True
        self._search_thread_pool[0]._running.add(trial_id)

        return config, prune_attr, 0

    def _get_ei_space(self):
        step_size = self._ls.STEPSIZE  # - EPS
        denorm_gs_admissible_min = self._ls.denormalize(
            {k: max(0, v - step_size) for k, v in self._gs_admissible_min.items()}
        )
        denorm_gs_admissible_max = self._ls.denormalize(
            {k: min(1, v + step_size) for k, v in self._gs_admissible_max.items()}
        )
        ei_space = deepcopy(self._search_thread_pool[0]._search_alg._space)
        for k in denorm_gs_admissible_min:
            ei_space[k].lower = denorm_gs_admissible_min[k]
            ei_space[k].upper = denorm_gs_admissible_max[k]
        return get_optuna_trial_suggestions(ei_space)

    def _suggest_from_global_search(
        self,
        trial_id: str,
        backup: int,
        retry: bool = True,
        iter: int = 0,
    ):
        if isinstance(
            self._search_thread_pool[0]._search_alg, ConditionalOptunaSearchCatBoost
        ):
            config = self._search_thread_pool[0].suggest(
                trial_id, reask=not retry, ei_space=self._get_ei_space()
            )
        else:
            config = self._search_thread_pool[0].suggest(trial_id, reask=not retry)
        prune_attr = config.get(self._ls.prune_attr, None) if config else None
        skip = self._should_skip(0, trial_id, config)
        if skip:
            estimator = None
            last_estimator_config = None
        else:
            estimator = config["Estimator"] if config else None
            last_estimator_config = config

        if not skip and self._valid(
            config,
        ):  # 1 + ((iter + 1) / self._MAX_GS_RETRIES)
            return config, prune_attr, 0, estimator
        else:
            config, prune_attr = None, None

        if retry:
            for i in range(self._MAX_GS_RETRIES):
                config, prune_attr = None, None
                config, prune_attr, _, _ = self._suggest_from_global_search(
                    trial_id, 1, retry=False, iter=i
                )
                if config and (estimator and config["Estimator"] != estimator):
                    pass
                elif config:
                    return config, prune_attr, 0, estimator

            last_estimator_config = last_estimator_config or config
            if last_estimator_config:
                estimator = last_estimator_config["Estimator"]
                self._mark_global_search_suggestion_as_an_error(trial_id)
                config, prune_attr, _ = self._force_suggestion_to_be_valid(
                    trial_id,
                    last_estimator_config,
                    0,  # step_multiplier=0.75
                )
                if self._should_skip(0, trial_id, config):
                    self._update_admissible_region(
                        config, self._gs_admissible_min, self._gs_admissible_max
                    )
                    return None, None, 0, estimator

        return config, prune_attr, 0, estimator

    def _mark_global_search_suggestion_as_an_error(self, trial_id: str):
        try:
            self._search_thread_pool[0]._search_alg.on_trial_complete(
                trial_id, result=None, state=TrialState.FAIL
            )
            print(f"Marked {trial_id} as an error")
        except Exception:
            print(f"Couldn't mark {trial_id} as an error, {traceback.format_exc()}")

    def _force_suggestion_to_be_valid(
        self,
        trial_id: str,
        config: dict,
        thread_id: int,
        step_multiplier=1,
    ):
        print("forcing config to be valid")
        config.pop(None, None)
        config = self._make_config_valid(config, step_multiplier=step_multiplier)
        if self._ls.prune_attr:
            config[self._ls.prune_attr] = self._ls.min_resource
        prune_attr = config.get(self._ls.prune_attr, None) if config else None

        return config, prune_attr, thread_id

    def _suggest_from_local_search(self, trial_id: str, thread_id: int):
        assert thread_id
        config = self._search_thread_pool[thread_id].suggest(trial_id)
        print(f"{trial_id} ls thread {thread_id} choice suggestion: {config}")
        prune_attr = config.get(self._ls.prune_attr, None) if config else None
        skip = self._should_skip(thread_id, trial_id, config)
        if skip:
            return None, None, thread_id

        return config, prune_attr, thread_id

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """choose thread, suggest a valid config"""
        print(f"suggest {trial_id}, {len(self._points_to_evaluate)}")
        prune_attr = None
        self._use_rs = False
        init = False
        if self._points_to_evaluate:
            init = True
            (
                config,
                prune_attr,
                proposing_thread,
            ) = self._suggest_from_points_to_evaluate(trial_id)

            config = self._clean_and_enforce_config(config, prune_attr)
            if not config:
                self._points_to_evaluate_len -= 1
                return None
        else:
            if not self._init_finished:
                choice = 0
                backup = 0
            else:
                choice, backup, local_threads_by_priority = self._select_thread()
                self._iters_without_new_best += 1
                if choice < 0:
                    print(f"skipping choice={choice}")
                    return None  # timeout
                elif choice:
                    if (
                        self._iters_without_new_best >= self._FORCE_GS_EVERY_N_ITER
                        and backup
                    ):
                        choice = 0
                        self._iters_without_new_best = 0
                else:
                    self._iters_without_new_best = 0
            print(
                f"{trial_id}: choice={choice}, backup={backup}, self._iters_without_new_best={self._iters_without_new_best}"
            )

            proposing_thread = choice

            if not choice:
                (
                    config,
                    prune_attr,
                    proposing_thread,
                    _,
                ) = self._suggest_from_global_search(
                    trial_id,
                    backup,
                )
                # new_backup = next(
                #     (
                #         thread_id
                #         for thread_id, thread, priority in local_threads_by_priority
                #         if thread.estimator == estimator
                #     ),
                #     None,
                # )
                # backup = new_backup if new_backup else backup
                if not config:
                    print("couldn't get a valid config from GS")
                    self._mark_global_search_suggestion_as_an_error(trial_id)
                    if choice != backup:
                        (
                            config,
                            prune_attr,
                            proposing_thread,
                        ) = self._suggest_from_local_search(trial_id, backup)
                    else:
                        return None

            else:
                config, prune_attr, proposing_thread = self._suggest_from_local_search(
                    trial_id, choice
                )

            if not config:
                if proposing_thread and proposing_thread in self._search_thread_pool:
                    # local search thread finishes
                    self._clean(proposing_thread)
                return None

            config = self._clean_and_enforce_config(config, prune_attr)
            if not config:
                return None

            if not proposing_thread:  # global search
                if self._ls._resource:
                    # TODO: add resource to config proposed by GS, min or median?
                    config[self._ls.prune_attr] = self._ls.min_resource
                    prune_attr = (
                        config.get(self._ls.prune_attr, None) if config else None
                    )
                # temporarily relax admissible region for parallel proposals
                self._update_admissible_region(
                    config, self._gs_admissible_min, self._gs_admissible_max
                )
            else:
                self._update_admissible_region(
                    config, self._ls_bound_min, self._ls_bound_max
                )
                print("\nGS BOUNDS UPDATE")
                print(self._ls.denormalize(self._ls_bound_min))
                print(self._ls.denormalize(self._ls_bound_max))
                print("")
                self._gs_admissible_min.update(self._ls_bound_min)
                self._gs_admissible_max.update(self._ls_bound_max)

        (
            result,
            config_signature,
        ) = self._has_config_been_already_tried(config)

        if result is not None:
            if init:
                self._points_to_evaluate_len -= 1
            return None

        self._result[config_signature] = {}
        self._suggested_configs[trial_id] = config
        self._trial_proposed_by[trial_id] = proposing_thread

        self._search_thread_pool[0]._search_alg.add_evaluated_trial(
            trial_id,
            config,
            state=TrialState.RUNNING,
        )

        # final_choice = self._trial_proposed_by.get(trial_id, 0)
        # if final_choice and self._reached_max_prune_attr:
        #     if final_choice not in self._diversification_multipliers:
        #         self._diversification_multipliers[final_choice] = 0
        #     else:
        #         self._diversification_multipliers[final_choice] += max((len(self._search_thread_pool)-1)/2, 1.5)
        # for k in self._diversification_multipliers.keys():
        #     self._diversification_multipliers[k] = max(
        #         self._diversification_multipliers[k]-0.5, 0
        #     )

        print(f"{trial_id} final suggestion by {proposing_thread}: {config}")
        config[META_KEY] = {
            "proposing_thread": proposing_thread,
            "init": init,
            **self._meta_info,
        }
        return config

    def _clean_and_enforce_config(self, config, prune_attr) -> dict:
        if prune_attr:
            config[self._ls.prune_attr] = prune_attr
            print(f"suggest prune_attr: {prune_attr}")
            assert prune_attr
            if prune_attr >= 1.0:
                self._reached_max_prune_attr = True
        try:
            clean_config = enforce_conditions_on_config(
                config, self._conditional_space_estimators
            )
            clean_config_len = len(clean_config)
            new_clean_config = enforce_conditions_on_config(
                config, self._conditional_space
            )
            assert (
                len(new_clean_config) >= clean_config_len
            ), f"{clean_config}, {new_clean_config}"
            clean_config = new_clean_config
            assert clean_config["Estimator"] == config["Estimator"]
        except:
            print(f"Bad configuration suggested, trying again")
            traceback.print_exc()
            print(self._conditional_space)
            print("")
            return None
        if prune_attr:
            clean_config[self._ls.prune_attr] = prune_attr
        clean_config = {k: numpy_to_python(v) for k, v in clean_config.items()}
        return clean_config

    def _has_config_been_already_tried(self, config) -> bool:
        config_signature = self._ls.config_signature(config)
        if not self._conditional_space:
            return (
                (self._result.get(config_signature, None) in self._result),
                config_signature,
                None,
            )
        result = None
        result = self._result.get(config_signature, None)
        return result, config_signature

    def _should_skip(self, choice, trial_id, config) -> bool:
        """if config is None or config's result is known or above mem threshold
        return True; o.w. return False
        """
        if config is None:
            return True
        (
            exists,
            config_signature,
        ) = self._has_config_been_already_tried(config)
        # check mem constraint
        if (
            exists is None
            and self._mem_threshold
            and self._mem_size(config) > self._mem_threshold
        ):
            self._result[config_signature] = {
                self._metric: np.inf * self._ls.metric_op,
                self._time_attr: 1,
            }
            exists = True
        if exists is not None:
            if not self._use_rs:
                result = self._result.get(config_signature)
                if result:
                    if choice:
                        self._search_thread_pool[choice].on_trial_complete(
                            trial_id, result, error=False, add_to_gs=False
                        )
                        # local search thread
                        self._clean(choice)
                # else:
                #    # tell the thread there is an error
                #    self._search_thread_pool[choice].on_trial_complete(
                #        trial_id, {}, error=True
                #    )
            return True
        return False

    def _clean(self, thread_id: int):
        """delete thread and increase admissible region if converged,
        merge local threads if they are close
        """
        assert thread_id
        assert thread_id in self._search_thread_pool
        todelete = set()
        for id in self._search_thread_pool:
            if id and id != thread_id:
                if self._inferior(id, thread_id):
                    print(f"thread {id} is inferior to {thread_id}")
                    todelete.add(id)
        for id in self._search_thread_pool:
            if id and id != thread_id:
                if self._inferior(thread_id, id):
                    print(f"thread {thread_id} is inferior to {id}")
                    todelete.add(thread_id)
                    break
        # logger.info(f"thead {thread_id}.converged="
        #     f"{self._search_thread_pool[thread_id].converged}")
        if self._search_thread_pool[thread_id].converged:
            todelete.add(thread_id)
            for key in self._ls_bound_max:
                if key in self._search_thread_pool[thread_id]._search_alg.space:
                    self._ls_bound_max[key] += self._ls.STEPSIZE
                    self._ls_bound_min[key] -= self._ls.STEPSIZE
        for id in todelete:
            print(f"THREAD CLEANER deleting {id}")
            del self._search_thread_pool[id]
            # self._diversification_multipliers.pop(id, None)


class BlendSearchTuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        use_extended: bool = True,
        num_samples: int = -1,
        time_budget_s: int = 600,
        secondary_pipeline_blueprint=None,
        target_metric=None,
        scoring=None,
        early_stopping: bool = True,
        cache=False,
        max_concurrent: int = 1,
        trainable_n_jobs: int = 4,
        display: Optional[IPythonDisplay] = None,
        stacking_level: int = 0,
        previous_stack=None,
        widget=None,
        plot_callback: Optional[BestPlotCallback] = None,
        **tune_kwargs,
    ) -> None:
        self.early_stopping = early_stopping
        super().__init__(
            problem_type=problem_type,
            pipeline_blueprint=pipeline_blueprint,
            cv=cv,
            random_state=random_state,
            use_extended=use_extended,
            num_samples=num_samples,
            cache=cache,
            time_budget_s=time_budget_s,
            target_metric=target_metric,
            scoring=scoring,
            display=display,
            max_concurrent=max_concurrent,
            trainable_n_jobs=trainable_n_jobs,
            secondary_pipeline_blueprint=secondary_pipeline_blueprint,
            stacking_level=stacking_level,
            widget=widget,
            plot_callback=plot_callback,
            previous_stack=previous_stack,
            **tune_kwargs,
        )
        self._searcher_kwargs = {
            "time_attr": "estimator_fit_time",
            "meta_info": {"stacking_level": self.stacking_level},
        }

    def _set_up_early_stopping(self, X, y, groups=None):
        step = 4
        if self.early_stopping and self.X_.shape[0] * self.X_.shape[1] > 100001:
            min_dist = self.cv.get_n_splits(self.X_, self.y_, self.groups_) * 1000
            if self.problem_type.is_classification():
                min_dist *= len(self.y_.cat.categories)
            min_dist /= self.X_.shape[0]
            if min_dist < 1:
                min_dist = max(min_dist, 10000 / self.X_.shape[0])

                self._searcher_kwargs["prune_attr"] = "dataset_fraction"
                self._searcher_kwargs["min_resource"] = np.around(min_dist, 2)
                self._searcher_kwargs["max_resource"] = 1.0
                self._searcher_kwargs["reduction_factor"] = step
                logger.debug(self._searcher_kwargs["prune_attr"])
        self.early_stopping_fractions_ = [1]

    def _add_extra_random_trials_to_default_grid(self):
        blend_search = ConditionalBlendSearch(
            space=self.pipeline_blueprint,
            metric="mean_validation_score",
            mode="max",
            points_to_evaluate=self.default_grid_,
            seed=self.random_state,
            use_extended=self.use_extended,
            **self._searcher_kwargs,
        )

        random_sampler = ConditionalOptunaSearch(
            space=self.pipeline_blueprint,
            metric="mean_validation_score",
            mode="max",
            sampler=RandomSampler(seed=self.random_state),
            seed=self.random_state,
            use_extended=self.use_extended,
        )

        init_config = blend_search._ls.init_config

        lower_bound = blend_search._ls.denormalize(
            {
                k: blend_search._gs_admissible_min[k] - blend_search._ls.STEPSIZE + 1e-8
                for k, v in init_config.items()
            }
        )
        upper_bound = blend_search._ls.denormalize(
            {
                k: blend_search._gs_admissible_min[k] + blend_search._ls.STEPSIZE - 1e-8
                for k, v in init_config.items()
            }
        )
        assert len(lower_bound) == len(upper_bound)

        def set_bounds(param, lower_bound, upper_bound):
            param_name = param[1][0]
            if param_name in lower_bound:
                return (
                    param[0],
                    (
                        param_name,
                        max(param[1][1], lower_bound[param_name]),
                        min(param[1][2], upper_bound[param_name]),
                    ),
                    param[2],
                )
            return param

        random_sampler._space = {
            k: set_bounds(v, lower_bound, upper_bound)
            for k, v in random_sampler._space.items()
        }

        # TODO move this into ConditionalBlendSearch, make it respect cost
        estimator_counts = collections.Counter()
        for config in self.default_grid_:
            estimator_counts[config["Estimator"]] += 1
        most_common_estimators = estimator_counts.most_common()
        target_count = int(
            np.median([count for estimator, count in most_common_estimators])
        )
        target_count = min(target_count, 10)
        extra_params = []

        for estimator, count in most_common_estimators:
            for _ in range(target_count - count):
                extra_params.append(
                    random_sampler.suggest(
                        1, reask=True, params={"Estimator": estimator}
                    )
                )

        del random_sampler

        self.default_grid_.extend(extra_params)

        self._remove_duplicates_from_grids()

    def _pre_search(self, X, y, X_test=None, y_test=None, groups=None):
        super()._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)

        # if self._cache:
        #    self._searcher_kwargs["time_attr"] = "estimator_fit_time"
        logger.debug(self._searcher_kwargs)

        # self._add_extra_random_trials_to_default_grid()

        # this is just to ensure constant order
        # TODO: make sure that extra trials are at the end
        # self._shuffle_default_grid()

        blend_search = ConditionalBlendSearch(
            space=self.secondary_pipeline_blueprint
            if self.secondary_pipeline_blueprint
            else self.pipeline_blueprint,
            metric="mean_validation_score",
            mode="max",
            points_to_evaluate=self.default_grid_,
            secondary_points_to_evaluate=self.secondary_grid_,
            seed=self.random_state,
            use_extended=self.use_extended,
            **self._searcher_kwargs,
        )
        blend_search._deadline = self._tune_kwargs["time_budget_s"] + time.time()

        self._tune_kwargs["search_alg"] = ConcurrencyLimiter(
            blend_search,
            max_concurrent=self.max_concurrent,
        )
