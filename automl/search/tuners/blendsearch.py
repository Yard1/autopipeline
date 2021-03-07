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
from typing import Dict, Optional, List, Tuple
import numpy as np
import traceback
import pickle
import os
import gc
import tempfile
from ray.tune.sample import Categorical

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter

from ray import tune
from ray.tune.suggest import Searcher, ConcurrencyLimiter
from ray.tune.suggest.variant_generator import generate_variants
from flaml.searcher.search_thread import SearchThread
from flaml.searcher.blendsearch import BlendSearch
from flaml.searcher.flow2 import FLOW2

import logging

logger = logging.getLogger(__name__)

from ray.tune.suggest import Searcher
from ray.tune import sample

from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

from .OptunaTPETuner import ConditionalOptunaSearch
from ..distributions import get_tune_distributions, CategoricalDistribution
from .utils import get_conditions, enforce_conditions_on_config, get_all_tunable_params
from .tuner import RayTuneTuner, remove_component_suffix
from ..utils import call_component_if_needed
from ...problems import ProblemType
from ...components import Component

GlobalSearch = ConditionalOptunaSearch


class PatchedFLOW2(FLOW2):
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
        self.signature_space = space
        self._random = np.random.RandomState(seed)
        self._seed = seed
        if not init_config:
            logger.warning(
                "No init config given to FLOW2. Using random initial config."
                "For cost-frugal search, "
                "consider providing init values for cost-related hps via "
                "'init_config'."
            )
        elif self.conditional_space:
            init_config = enforce_conditions_on_config(init_config, conditional_space)
            print(f"FLOW2 init_config {init_config}")
        self.init_config = self.best_config = init_config
        if limit_space_to_init_config:
            assert init_config
            self.space = {
                k: v if Component._automl_id_sign in k else init_config[k]
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
        if space:
            self._init_search()

    def config_signature(self, config) -> tuple:
        """return the signature tuple of a config"""
        value_list = []
        for key in self.signature_space.keys():
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

    @property
    def step_lower_bound(self) -> float:
        step_lb = self._step_lb
        for key in self._tunable_keys:
            domain = self.space[key]
            sampler = domain.get_sampler()
            if isinstance(sampler, sample.Quantized):
                sampler_inner = sampler.get_sampler()
                if str(sampler_inner) == "LogUniform":
                    if key in self.best_config:
                        step_lb = min(
                            step_lb,
                            np.log(1.0 + sampler.q / self.best_config[key])
                            / np.log(domain.upper / domain.lower),
                        )
            elif isinstance(domain, sample.Integer) and str(sampler) == "LogUniform":
                if key in self.best_config:
                    step_lb = min(
                        step_lb,
                        np.log(1.0 + 1.0 / self.best_config[key])
                        / np.log(domain.upper / domain.lower),
                    )
        if np.isinf(step_lb):
            step_lb = self.STEP_LOWER_BOUND
        else:
            step_lb *= np.sqrt(self.dim)
        return step_lb

    def create(
        self,
        init_config: Dict,
        obj: float,
        cost: float,
        conditional_space=None,
        limit_space_to_init_config: bool = False,
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
            self._seed + 1,
            conditional_space=conditional_space,
            limit_space_to_init_config=limit_space_to_init_config,
        )
        flow2.best_obj = obj * self.metric_op  # minimize internally
        flow2.cost_incumbent = cost
        return flow2

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        """compare with incumbent"""
        # if better, move, reset num_complete and num_proposed
        # if not better and num_complete >= 2*dim, num_allowed += 2
        self.trial_count += 1
        print("flow2 on_trial_complete")
        print(result)
        print(self.space)
        if not error and result:
            obj = result.get(self._metric)
            if obj:
                obj *= self.metric_op
                if obj < self.best_obj:
                    self.best_obj, self.best_config = obj, self._configs[trial_id]
                    self.incumbent = self.normalize(self.best_config)
                    self.cost_incumbent = result.get(self.cost_attr)
                    if self._resource:
                        self._resource = self.best_config[self.prune_attr]
                    self._num_complete4incumbent = 0
                    self._cost_complete4incumbent = 0
                    self._num_allowed4incumbent = 2 * self.dim
                    self._proposed_by.clear()
                    if self._K > 0:
                        self.step *= np.sqrt(self._K / self._oldK)
                    if self.step > self.step_ub:
                        self.step = self.step_ub
                    self._iter_best_config = self.trial_count
                    return
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
                if self.step >= self.step_lower_bound:
                    # decrease step size
                    self._oldK = self._K if self._K else self._iter_best_config
                    self._K = self.trial_count + 1
                    self.step *= np.sqrt(self._oldK / self._K)
                    # logger.info(f"step={self.step}, lb={self.step_lower_bound}")
                self._num_complete4incumbent -= 2
                if self._num_allowed4incumbent < 2:
                    self._num_allowed4incumbent = 2
        # elif proposed_by: # proposed by older incumbent
        #     del self._proposed_by[trial_id]


LocalSearch = PatchedFLOW2


class SharingSearchThread(SearchThread):
    """Class of global or local search thread"""

    def __init__(
        self,
        mode: str = "min",
        search_alg: Optional[Searcher] = None,
        cost_attr=None,
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
        self._init_config = True
        if cost_attr:
            self.cost_attr = cost_attr
        self.prune_attr = prune_attr
        if prune_attr:
            assert max_prune_attr
        self.max_prune_attr = max_prune_attr

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        """update the statistics of the thread"""
        print(f"on_trial_complete {trial_id} {self._search_alg}")
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials"):
            # optuna doesn't handle error
            if self._is_ls or not self._init_config:
                self._search_alg.on_trial_complete(
                    trial_id,
                    enforce_conditions_on_config(
                        result,
                        self._search_alg.conditional_space,
                        prefix="config/",
                        keys_to_keep=set(self._search_alg.space).union(
                            {self.prune_attr, self.cost_attr}
                        ),
                    ),
                    error,
                )
            else:
                # init config is not proposed by self._search_alg
                # under this thread
                self._init_config = False
        elif not error:
            print(
                f"Optuna has {len(self._search_alg._ot_study.trials)} trials in memory"
            )
            try:
                if trial_id in self._search_alg._ot_trials:
                    if (
                        f"{self.prune_attr}_" not in result
                        or np.around(result[f"{self.prune_attr}_"], 1)
                        >= self.max_prune_attr
                    ):
                        print("adding trial result to optuna")
                        self._search_alg.on_trial_complete(
                            trial_id,
                            enforce_conditions_on_config(
                                result,
                                self._search_alg._conditional_space,
                                prefix="config/",
                                keys_to_keep=self._search_alg._space,
                            ),
                        )
                    else:
                        self._search_alg.on_trial_result(
                            trial_id,
                            enforce_conditions_on_config(
                                result,
                                self._search_alg._conditional_space,
                                prefix="config/",
                                keys_to_keep=self._search_alg._space,
                            ),
                        )
                elif (
                    f"{self.prune_attr}_" not in result
                    or np.around(result[f"{self.prune_attr}_"], 1)
                    >= self.max_prune_attr
                ):
                    print("adding ls result to optuna")
                    return_val = self._search_alg.add_evaluated_trial(
                        trial_id,
                        enforce_conditions_on_config(
                            result,
                            self._search_alg._conditional_space,
                            prefix="config/",
                            keys_to_keep=self._search_alg._space,
                        ),
                    )
                    assert return_val
            except:
                print(
                    f"couldn't add trial {result} to optuna.\n{traceback.format_exc()}"
                )

        if result:
            if self.cost_attr in result:
                self.cost_last = result[self.cost_attr]
                self.cost_total += self.cost_last
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

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """use the suggest() of the underlying search algorithm"""
        if isinstance(self._search_alg, FLOW2):
            config = self._search_alg.suggest(trial_id)
        else:
            # config = self._search_alg.suggest(trial_id)
            try:
                config = self._search_alg.suggest(trial_id)
            except:
                logger.warning(
                    "The global search method raises error. "
                    "Ignoring for this iteration.\n"
                    f"{traceback.format_exc()}"
                )
                config = None
        return config

    def on_trial_result(self, trial_id: str, result: Dict):
        """TODO update the statistics of the thread with partial result?"""
        # print('[SearchThread] on trial result')
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
                enforce_conditions_on_config(
                    result,
                    self._search_alg._conditional_space,
                    prefix="config/",
                    keys_to_keep=self._search_alg._space,
                ),
            )
        if self.cost_attr in result and self.cost_last < result[self.cost_attr]:
            self.cost_last = result[self.cost_attr]
            # self._update_speed()


class ConditionalBlendSearch(BlendSearch):
    """class for BlendSearch algorithm"""

    _force_gs_after = 100

    def __init__(
        self,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        space: Optional[dict] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        cat_hp_cost: Optional[dict] = None,
        prune_attr: Optional[str] = None,
        min_resource: Optional[float] = None,
        max_resource: Optional[float] = None,
        reduction_factor: Optional[float] = None,
        resources_per_trial: Optional[dict] = None,
        global_search_alg: Optional[Searcher] = None,
        time_attr: Optional[str] = None,
        seed: Optional[int] = None,
        mem_size=None,
    ):
        """Constructor

        Args:
            metric: A string of the metric name to optimize for.
                minimization or maximization.
            mode: A string in ['min', 'max'] to specify the objective as
            space: A dictionary to specify the search space.
            points_to_evaluate: Initial parameter suggestions to be run first.
                The first element needs to be a dictionary from a subset of
                controlled dimensions to the initial low-cost values.
                e.g.,

                .. code-block:: python

                    [{'epochs': 1}]

            cat_hp_cost: A dictionary from a subset of categorical dimensions
                to the relative cost of each choice.
                e.g.,

                .. code-block:: python

                    {'tree_method': [1, 1, 2]}

                i.e., the relative cost of the
                three choices of 'tree_method' is 1, 1 and 2 respectively.
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
            reduction_factor: A float of the reduction factor used for
                incremental pruning.
            resources_per_trial: A dictionary of the resources permitted per
                trial, such as 'mem'.
            global_search_alg: A Searcher instance as the global search
                instance. If omitted, Optuna is used. The following algos have
                known issues when used as global_search_alg:
                - HyperOptSearch raises exception sometimes
                - TuneBOHB has its own scheduler
            mem_size: A function to estimate the memory size for a given config.
        """
        self._metric, self._mode = metric, mode
        self._conditional_space = get_conditions(space, to_str=True)
        self._time_attr = "time_total_s" or time_attr

        LocalSearch.cost_attr = self._time_attr
        self._seed = seed

        if global_search_alg is not None:
            self._gs = global_search_alg
        elif getattr(self, "__name__", None) != "CFO":
            self._gs = GlobalSearch(
                space=space,
                metric=metric,
                mode=mode,
                seed=seed,
                n_startup_trials=1,
            )
        else:
            self._gs = None

        init_config = self._get_all_default_values(space, get_categorical=False)
        space, _ = get_all_tunable_params(space, to_str=True)
        space = get_tune_distributions(space)
        const_values = {
            k
            for k, v in space.items()
            if isinstance(v, Categorical) and len(v.categories) == 1
        }
        space = {k: v for k, v in space.items() if k not in const_values}
        if points_to_evaluate:
            points_to_evaluate = [
                {k: v for k, v in point.items() if k in space}
                for point in points_to_evaluate
            ]
        self._points_to_evaluate = points_to_evaluate
        self._ls = LocalSearch(
            init_config,
            metric,
            mode,
            cat_hp_cost,
            space,
            prune_attr,
            min_resource,
            max_resource,
            reduction_factor,
            seed,
        )
        self._ls.cost_attr = self._time_attr
        self._resources_per_trial = resources_per_trial
        self._mem_size = mem_size
        self._mem_threshold = (
            resources_per_trial.get("mem") if resources_per_trial else None
        )
        self._init_search()

    @property
    def _conditional_space_estimators(self):
        return {"Estimator": self._conditional_space["Estimator"]}

    def _get_all_default_values(self, pipeline_blueprint, get_categorical=True) -> dict:
        default_grid = {
            k: v for k, v in pipeline_blueprint.get_all_distributions().items()
        }
        default_values = {}
        for k, v in default_grid.items():
            for v2 in v.values:
                for k3, v3 in v2.get_tuning_grid().items():
                    if not get_categorical and isinstance(v3, CategoricalDistribution):
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
        self._last_global_search = 0

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
            self._last_global_search,
        )
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
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
            self._last_global_search,
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
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        """search thread updater and cleaner"""
        thread_id = self._trial_proposed_by.get(trial_id)
        if thread_id in self._search_thread_pool:
            self._search_thread_pool[thread_id].on_trial_complete(
                trial_id, result, error
            )
            if thread_id > 0 and self._global_search_thread:
                self._global_search_thread.on_trial_complete(trial_id, result, error)
            del self._trial_proposed_by[trial_id]
            # if not thread_id: logger.info(f"result {result}")
        if result:
            config = self._suggested_configs[trial_id]
            (
                _,
                config_signature,
                enforced_config_signature,
            ) = self._has_config_been_already_tried(config)
            if error:  # remove from result cache
                del self._result[config_signature]
            else:  # add to result cache
                self._result[config_signature] = result
                if enforced_config_signature:
                    self._result[enforced_config_signature] = result
            # update target metric if improved
            if (result[self._metric] - self._metric_target) * self._ls.metric_op < 0:
                self._metric_target = result[self._metric]
            if not thread_id and self._create_condition(result):
                # thread creator
                self._search_thread_pool[self._thread_count] = SharingSearchThread(
                    self._ls.mode,
                    self._ls.create(
                        config,
                        result[self._metric],
                        cost=result[self._time_attr],
                        conditional_space=self._conditional_space,
                        limit_space_to_init_config=True,
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
            # reset admissible region to ls bounding box
            self._gs_admissible_min.update(self._ls_bound_min)
            self._gs_admissible_max.update(self._ls_bound_max)

        # cleaner
        # logger.info(f"thread {thread_id} in search thread pool="
        #     f"{thread_id in self._search_thread_pool}")
        if thread_id and thread_id in self._search_thread_pool:
            # local search thread
            self._clean(thread_id)

    def _update_admissible_region(self, config, admissible_min, admissible_max):
        # update admissible region
        normalized_config = self._ls.normalize(config)
        for key in admissible_min:
            if key in config:
                value = normalized_config[key]
                if value > admissible_max[key]:
                    admissible_max[key] = value
                elif value < admissible_min[key]:
                    admissible_min[key] = value

    def _valid(self, config: Dict) -> bool:
        """config validator"""
        try:
            for key in self._gs_admissible_min:
                if key in config:
                    value = config[key]
                    # logger.info(
                    #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                    if (
                        value + self._ls.STEPSIZE < self._gs_admissible_min[key]
                        or value > self._gs_admissible_max[key] + self._ls.STEPSIZE
                    ):
                        return False
        except TypeError:
            normalized_config = self._ls.normalize(config)
            for key in self._gs_admissible_min:
                if key in config:
                    value = normalized_config[key]
                    # logger.info(
                    #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                    if (
                        value + self._ls.STEPSIZE < self._gs_admissible_min[key]
                        or value > self._gs_admissible_max[key] + self._ls.STEPSIZE
                    ):
                        return False
        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """choose thread, suggest a valid config"""
        print(f"suggest {trial_id}")
        if self._init_used and not self._points_to_evaluate:
            choice, backup = self._select_thread()
            if choice < 0:
                return None  # timeout
            elif choice:
                if self._last_global_search >= self._force_gs_after and backup:
                    choice = 0
                else:
                    self._last_global_search += 1
            if not choice:
                self._last_global_search = 0
            print(
                f"{trial_id}: choice={choice}, backup={backup}, self._last_global_search={self._last_global_search}"
            )
            self._use_rs = False
            config = self._search_thread_pool[choice].suggest(trial_id)
            print(f"main choice suggestion: {config}")
            skip = self._should_skip(choice, trial_id, config)
            if skip:
                if choice:
                    # logger.info(f"skipping choice={choice}, config={config}")
                    return None
                # use rs
                self._use_rs = True
                for _, generated in generate_variants({"config": self._ls.space}):
                    config = {**config, **generated["config"]}
                    break
                # logger.debug(f"random config {config}")
                skip = self._should_skip(choice, trial_id, config)
                if skip:
                    return None
            # if not choice: logger.info(config)
            if choice or self._valid(config):
                # LS or valid or no backup choice
                self._trial_proposed_by[trial_id] = choice
            else:  # invalid config proposed by GS
                # if not self._use_rs:
                #    self._search_thread_pool[choice].on_trial_complete(
                #        trial_id, {}, error=True
                #    )  # tell GS there is an error
                self._use_rs = False
                if choice == backup:
                    # use CFO's init point
                    init_config = self._ls.init_config
                    config = self._ls.complete_config(
                        init_config, self._ls_bound_min, self._ls_bound_max
                    )
                    self._trial_proposed_by[trial_id] = choice
                else:
                    config = self._search_thread_pool[backup].suggest(trial_id)
                    skip = self._should_skip(backup, trial_id, config)
                    if skip:
                        return None
                    self._trial_proposed_by[trial_id] = backup
                    choice = backup
            if not choice:  # global search
                if self._ls._resource:
                    # TODO: add resource to config proposed by GS, min or median?
                    config[self._ls.prune_attr] = self._ls.min_resource
                # temporarily relax admissible region for parallel proposals
                self._update_admissible_region(
                    config, self._gs_admissible_min, self._gs_admissible_max
                )
            else:
                self._update_admissible_region(
                    config, self._ls_bound_min, self._ls_bound_max
                )
                self._gs_admissible_min.update(self._ls_bound_min)
                self._gs_admissible_max.update(self._ls_bound_max)
            (
                result,
                config_signature,
                enforced_config_signature,
            ) = self._has_config_been_already_tried(config)
            self._result[config_signature] = {}
            if enforced_config_signature:
                self._result[enforced_config_signature] = {}
        else:  # use init config
            init_config = (
                self._points_to_evaluate.pop(0)
                if self._points_to_evaluate
                else self._ls.init_config
            )
            config = self._ls.complete_config(
                init_config, self._ls_bound_min, self._ls_bound_max
            )
            assert len(config) == len(self._ls.space) + int(bool(self._ls.prune_attr))
            # logger.info(f"reset config to {config}")
            (
                result,
                config_signature,
                enforced_config_signature,
            ) = self._has_config_been_already_tried(config)
            if result:  # tried before
                # self.on_trial_complete(trial_id, result)
                return None
            elif result is None:  # not tried before
                self._result[config_signature] = {}
                if enforced_config_signature:
                    self._result[enforced_config_signature] = {}
            else:
                return None  # running but no result yet
            self._init_used = True
            self._trial_proposed_by[trial_id] = 0
        # logger.info(f"config={config}")
        self._suggested_configs[trial_id] = config
        if self._ls.prune_attr:
            prune_attr = config[self._ls.prune_attr]
            print(f"suggest prune_attr: {prune_attr}")
            assert prune_attr
        clean_config = {}
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
        except:
            print("Bad configuration suggested, trying again")
            traceback.print_exc()
            print(self._conditional_space)
            print("")
            return None
        if self._ls.prune_attr:
            clean_config[self._ls.prune_attr] = prune_attr
        return clean_config

    def _has_config_been_already_tried(self, config) -> bool:
        config_signature = self._ls.config_signature(config)
        if not self._conditional_space:
            return (
                (self._result.get(config_signature, None) in self._result),
                config_signature,
                None,
            )
        enforced_config_signature = self._ls.config_signature(
            enforce_conditions_on_config(
                config, self._conditional_space, raise_exceptions=False
            )
        )
        result = None
        result = self._result.get(config_signature, None)
        if not result:
            result = self._result.get(enforced_config_signature, None)
        return result, config_signature, enforced_config_signature

    def _should_skip(self, choice, trial_id, config) -> bool:
        """if config is None or config's result is known or above mem threshold
        return True; o.w. return False
        """
        if config is None:
            return True
        (
            exists,
            config_signature,
            enforced_config_signature,
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
                if not result and enforced_config_signature:
                    result = self._result.get(enforced_config_signature)
                if result:
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, result, error=False
                    )
                    if choice:
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
        todelete = set()
        for id in self._search_thread_pool:
            if id and id != thread_id:
                if self._inferior(id, thread_id):
                    todelete.add(id)
        for id in self._search_thread_pool:
            if id and id != thread_id:
                if self._inferior(thread_id, id):
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
            del self._search_thread_pool[id]


class BlendSearchTuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        num_samples: int = 100,
        early_stopping=True,
        cache=False,
        **tune_kwargs,
    ) -> None:
        self.early_stopping = early_stopping
        super().__init__(
            problem_type=problem_type,
            pipeline_blueprint=pipeline_blueprint,
            cv=cv,
            random_state=random_state,
            num_samples=num_samples,
            cache=cache,
            **tune_kwargs,
        )
        self._searcher_kwargs = {}

    def _set_up_early_stopping(self, X, y, groups=None):
        if self.early_stopping and self.X_.shape[0] > 20000:
            min_dist = self.cv.get_n_splits(self.X_, self.y_, self.groups_) * 20
            if self.problem_type.is_classification():
                min_dist *= len(self.y_.cat.categories)
            min_dist /= self.X_.shape[0]
            min_dist = max(min_dist, 10000 / self.X_.shape[0])

            step = 4

            self._searcher_kwargs["prune_attr"] = "dataset_fraction"
            self._searcher_kwargs["min_resource"] = min_dist
            self._searcher_kwargs["max_resource"] = 1.0
            self._searcher_kwargs["reduction_factor"] = step
            print(self._searcher_kwargs["prune_attr"])
        else:
            self.early_stopping_fractions_ = [1]

    def _pre_search(self, X, y, groups=None):
        super()._pre_search(X, y, groups=groups)
        if self._cache:
            self._searcher_kwargs["time_attr"] = "estimator_fit_time"
        self._tune_kwargs["search_alg"] = ConditionalBlendSearch(
            space=self.pipeline_blueprint,
            metric="mean_test_score",
            mode="max",
            points_to_evaluate=self.default_grid,
            seed=self.random_state,
            **self._searcher_kwargs,
        )

    def _trial_with_cv(self, config, checkpoint_dir=None):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        prune_attr = self._searcher_kwargs.get("prune_attr")

        config_called = self._treat_config(config)
        config_called.pop(prune_attr, None)

        if prune_attr:
            prune_attr = config.get(prune_attr)

        print(f"prune_attr: {prune_attr}")

        print(config_called)

        estimator.set_params(**config_called)
        memory = tempfile.gettempdir() if self._cache is True else self._cache
        memory = memory if not memory == os.getcwd() else ".."
        estimator.set_params(memory=memory)

        if prune_attr and prune_attr < 1.0:
            subsample_cv = _SubsampleMetaSplitter(
                base_cv=self.cv,
                fraction=prune_attr,
                subsample_test=True,
                random_state=self.random_state,
            )
        else:
            subsample_cv = self.cv

        scores = cross_validate(
            estimator,
            self.X_,
            self.y_,
            cv=subsample_cv,
            groups=self.groups_,
            error_score="raise",
            return_estimator=True,
            verbose=0,
            # fit_params=self.fit_params,
            # groups=self.groups,
            # return_train_score=self.return_train_score,
            # scoring=self.scoring,
        )

        estimator_fit_time = np.sum(
            [x.final_estimator_fit_time_ for x in scores["estimator"]]
        )
        gc.collect()
        if prune_attr:
            tune.report(
                done=True,
                mean_test_score=np.mean(scores["test_score"]),
                dataset_fraction_=prune_attr,
                estimator_fit_time=estimator_fit_time,
            )
        else:
            tune.report(
                done=True,
                mean_test_score=np.mean(scores["test_score"]),
                estimator_fit_time=estimator_fit_time,
            )

    def _search(self, X, y, groups=None):
        self._pre_search(X, y, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, groups=None):
        return self._search(X, y, groups=groups)
