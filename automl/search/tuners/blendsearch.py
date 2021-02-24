"""!
 * Copyright (c) 2020-2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the
 * project root for license information.
"""
from typing import Dict, Optional, List, Tuple
import numpy as np
import time
import pickle
from ray.tune.suggest import Searcher
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.variant_generator import generate_variants
from .flow2 import FLOW2 as LocalSearch

import logging

logger = logging.getLogger(__name__)

from ray.tune.suggest import Searcher
from .flow2 import FLOW2

from ray.tune.result import DEFAULT_METRIC, TRAINING_ITERATION
from ray.tune.sample import Categorical, Domain, Float, Integer, LogUniform, \
    Quantized, Uniform
from ray.tune.suggest.suggestion import UNRESOLVED_SEARCH_SPACE, \
    UNDEFINED_METRIC_MODE, UNDEFINED_SEARCH_SPACE
from ray.tune.suggest.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict

import optuna as ot
from optuna.samplers import BaseSampler

conditional_space = {"cond_param1": ("cond_param2", [False])}

def enforce_conditions_on_config(config, prefix=""):
    config = config.copy()
    for independent_name, v in conditional_space.items():
        dependent_name, required_values = v
        if config[f"{prefix}{independent_name}"] not in required_values and f"{prefix}{dependent_name}" in config:
            config.pop(f"{prefix}{dependent_name}", None)
    return config

class OptunaConditional(OptunaSearch):
    def suggest(self, trial_id: str) -> Optional[Dict]:
        if not self._space:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"))
        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__,
                    metric=self._metric,
                    mode=self._mode))

        if trial_id not in self._ot_trials:
            ot_trial_id = self._storage.create_new_trial(
                self._ot_study._study_id)
            self._ot_trials[trial_id] = ot.trial.Trial(self._ot_study,
                                                       ot_trial_id)
        ot_trial = self._ot_trials[trial_id]

        if self._points_to_evaluate:
            params = self._points_to_evaluate.pop(0)
        else:
            # getattr will fetch the trial.suggest_ function on Optuna trials
            params = self._get_params(ot_trial)
        return unflatten_dict(params)

    def _get_name(self, args, kwargs):
        return args[0] if len(args) > 0 else kwargs["name"]

    def _get_optuna_trial_value(self, ot_trial, tpl):
        fn, args, kwargs = tpl
        return getattr(ot_trial, fn)(*args, **kwargs)

    def _get_params(self, ot_trial):
        params_checked = set()
        params = {}
        space_dict = {
            self._get_name(args, kwargs): (fn, args, kwargs) for fn, args, kwargs in self._space
        }
        for key, condition in conditional_space.items():
            value = self._get_optuna_trial_value(ot_trial, space_dict[key])
            params[key] = value
            params_checked.add(key)
            params_checked.add(condition[0])
            if value in condition[1]:
                params[condition[0]] = self._get_optuna_trial_value(ot_trial, space_dict[condition[0]])

        for key, tpl in space_dict.items():
            if key in params_checked:
                continue
            value = self._get_optuna_trial_value(ot_trial, space_dict[key])
            params[key] = value
        return params

GlobalSearch = OptunaConditional


class SearchThread:
    """Class of global or local search thread"""

    cost_attr = "time_total_s"

    def __init__(self, mode: str = "min", search_alg: Optional[Searcher] = None):
        """When search_alg is omitted, use local search FLOW2"""
        self._search_alg = search_alg
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

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """use the suggest() of the underlying search algorithm"""
        if isinstance(self._search_alg, FLOW2):
            config = self._search_alg.suggest(trial_id)
        else:
            try:
                config = self._search_alg.suggest(trial_id)
            except:
                logger.warning(
                    f"The global search method raises error. "
                    "Ignoring for this iteration."
                )
                config = None
        return config

    def update_priority(self, eci: Optional[float] = 0):
        # optimistic projection
        self.priority = eci * self.speed - self.obj_best1

    def update_eci(self, metric_target: float, max_speed: Optional[float] = np.inf):
        # calculate eci: expected cost for improvement over metric_target;
        best_obj = metric_target * self._metric_op
        if not self.speed:
            self.speed = max_speed
        self.eci = max(
            self.cost_total - self.cost_best1, self.cost_best1 - self.cost_best2
        )
        if self.obj_best1 > best_obj and self.speed > 0:
            self.eci = max(self.eci, 2 * (self.obj_best1 - best_obj) / self.speed)

    def _update_speed(self):
        # calculate speed; use 0 for invalid speed temporarily
        if self.obj_best2 > self.obj_best1:
            self.speed = (self.obj_best2 - self.obj_best1) / (
                self.cost_total - self.cost_best2
            )
        else:
            self.speed = 0

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        """update the statistics of the thread"""
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials") or (
            not error and trial_id in self._search_alg._ot_trials
        ):
            # optuna doesn't handle error
            self._search_alg.on_trial_complete(trial_id, enforce_conditions_on_config(result, prefix="config/"), error)
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

    def on_trial_result(self, trial_id: str, result: Dict):
        """TODO update the statistics of the thread with partial result?"""
        # print('[SearchThread] on trial result')
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials") or (
            trial_id in self._search_alg._ot_trials
        ):
            self._search_alg.on_trial_result(trial_id, result)
        if self.cost_attr in result and self.cost_last < result[self.cost_attr]:
            self.cost_last = result[self.cost_attr]
            # self._update_speed()

    @property
    def converged(self) -> bool:
        return self._search_alg.converged

    @property
    def resource(self) -> float:
        return self._search_alg.resource

    def reach(self, thread) -> bool:
        """whether the incumbent can reach the incumbent of thread"""
        return self._search_alg.reach(thread._search_alg)

    @property
    def can_suggest(self) -> bool:
        """whether the thread can suggest new configs"""
        return self._search_alg.can_suggest


class BlendSearch(Searcher):
    '''class for BlendSearch algorithm
    '''

    def __init__(self,
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
                 mem_size = None):
        '''Constructor
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
        '''
        self._metric, self._mode = metric, mode
        if points_to_evaluate: init_config = points_to_evaluate[0]
        else: init_config = {}
        self._points_to_evaluate = points_to_evaluate
        if global_search_alg is not None:
            self._gs = global_search_alg
        elif getattr(self, '__name__', None) != 'CFO':
            self._gs = GlobalSearch(space=space, metric=metric, mode=mode)
        else:
            self._gs = None
        self._ls = LocalSearch(init_config, metric, mode, cat_hp_cost, space,
         prune_attr, min_resource, max_resource, reduction_factor)
        self._resources_per_trial = resources_per_trial
        self._mem_size = mem_size
        self._mem_threshold = resources_per_trial.get(
            'mem') if resources_per_trial else None
        self._init_search()
        
    def set_search_properties(self,
                              metric: Optional[str] = None,
                              mode: Optional[str] = None,
                              config: Optional[Dict] = None) -> bool:
        if self._ls.space:
            if 'time_budget_s' in config:
                self._deadline = config.get('time_budget_s') + time.time()
            if 'metric_target' in config:
                self._metric_target = config.get('metric_target')
        else:
            if metric: self._metric = metric
            if mode: self._mode = mode
            self._ls.set_search_properties(metric, mode, config)
            if self._gs is not None:
                self._gs.set_search_properties(metric, mode, config)
            self._init_search()
        return True

    def _init_search(self):
        '''initialize the search
        '''
        self._metric_target = np.inf * self._ls.metric_op
        self._search_thread_pool = {
            # id: int -> thread: SearchThread
            0: SearchThread(self._ls.mode, self._gs)
            } 
        self._thread_count = 1 # total # threads created
        self._init_used = self._ls.init_config is None
        self._trial_proposed_by = {} # trial_id: str -> thread_id: int
        self._admissible_min = self._ls.normalize(self._ls.init_config)
        self._admissible_max = self._admissible_min.copy()
        self._result = {} # config_signature: tuple -> result: Dict
        self._config_results = {}
        self._deadline = np.inf

    def save(self, checkpoint_path: str):
        save_object = (self._metric_target, self._search_thread_pool,
            self._thread_count, self._init_used, self._trial_proposed_by,
            self._admissible_min, self._admissible_max, self._result,
            self._deadline)
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)
            
    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
        self._metric_target, self._search_thread_pool, \
            self._thread_count, self._init_used, self._trial_proposed_by, \
            self._admissible_min, self._admissible_max, self._result, \
            self._deadline = save_object

    def restore_from_dir(self, checkpoint_dir: str):
        super.restore_from_dir(checkpoint_dir)

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None,
                          error: bool = False):
        ''' search thread updater and cleaner
        '''
        print("######################")
        print("on_trial_complete")
        thread_id = self._trial_proposed_by.get(trial_id)
        if thread_id in self._search_thread_pool: 
            self._search_thread_pool[thread_id].on_trial_complete(
            trial_id, result, error)
            del self._trial_proposed_by[trial_id]
            # if not thread_id: logger.info(f"result {result}")
        print(thread_id)
        if result:
            config = {**self._ls.best_config, **self._config_results[trial_id]}
            print(config)
            if error: # remove from result cache
                del self._result[self._ls.config_signature(config)]
            else: # add to result cache
                self._result[self._ls.config_signature(config)] = result
            # update target metric if improved
            if (result[self._metric]-self._metric_target)*self._ls.metric_op<0:
                self._metric_target = result[self._metric]
            if thread_id: # from local search
                # update admissible region
                normalized_config = self._ls.normalize(config)
                for key in self._admissible_min:
                    value = normalized_config[key]
                    if value > self._admissible_max[key]:
                        self._admissible_max[key] = value
                    elif value < self._admissible_min[key]:
                        self._admissible_min[key] = value
            elif self._create_condition(result):
                # thread creator
                self._search_thread_pool[self._thread_count] = SearchThread(
                    self._ls.mode,
                    self._ls.create(config, result[self._metric], cost=result[
                        "time_total_s"])
                )
                thread_id = self._thread_count
                self._thread_count += 1
        print("######################")
        # cleaner
        # logger.info(f"thread {thread_id} in search thread pool="
        #     f"{thread_id in self._search_thread_pool}")
        if thread_id and thread_id in self._search_thread_pool:
            # local search thread
            self._clean(thread_id)

    def _create_condition(self, result: Dict) -> bool:
        ''' create thread condition
        '''
        if len(self._search_thread_pool) < 2: return True
        obj_median = np.median([thread.obj_best1 for id, thread in
         self._search_thread_pool.items() if id])
        return result[self._metric] * self._ls.metric_op < obj_median

    def _clean(self, thread_id: int):
        ''' delete thread and increase admissible region if converged,
        merge local threads if they are close
        '''
        assert thread_id
        todelete = set()
        for id in self._search_thread_pool:
            if id and id!=thread_id:
                if self._inferior(id, thread_id):
                    todelete.add(id)
        for id in self._search_thread_pool:
            if id and id!=thread_id:
                if self._inferior(thread_id, id):
                    todelete.add(thread_id)
                    break        
        # logger.info(f"thead {thread_id}.converged="
        #     f"{self._search_thread_pool[thread_id].converged}")
        if self._search_thread_pool[thread_id].converged:
            todelete.add(thread_id)
            for key in self._admissible_min:
                self._admissible_max[key] += self._ls.STEPSIZE
                self._admissible_min[key] -= self._ls.STEPSIZE            
        for id in todelete:
            del self._search_thread_pool[id]

    def _inferior(self, id1: int, id2: int) -> bool:
        ''' whether thread id1 is inferior to id2
        '''
        t1 = self._search_thread_pool[id1]
        t2 = self._search_thread_pool[id2]
        if t1.obj_best1 < t2.obj_best2: return False
        elif t1.resource and t1.resource < t2.resource: return False
        elif t2.reach(t1): return True
        else: return False

    def on_trial_result(self, trial_id: str, result: Dict):
        if trial_id not in self._trial_proposed_by: return
        thread_id = self._trial_proposed_by[trial_id]
        if not thread_id in self._search_thread_pool: return
        self._search_thread_pool[thread_id].on_trial_result(trial_id, result)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        ''' choose thread, suggest a valid config
        '''
        if self._init_used and not self._points_to_evaluate:
            choice, backup = self._select_thread()
            # logger.debug(f"choice={choice}, backup={backup}")
            if choice < 0: return None # timeout
            self._use_rs = False
            config = self._search_thread_pool[choice].suggest(trial_id)
            skip = self._should_skip(choice, trial_id, config)
            if skip:
                if choice: 
                    # logger.info(f"skipping choice={choice}, config={config}")
                    return None
                # use rs
                self._use_rs = True
                for _, generated in generate_variants(
                    {'config': self._ls.space}):
                    config = generated['config']
                    break
                # logger.debug(f"random config {config}")
                skip = self._should_skip(choice, trial_id, config)
                if skip: return None
            # if not choice: logger.info(config)
            if choice or backup == choice or self._valid(config): 
                # LS or valid or no backup choice
                self._trial_proposed_by[trial_id] = choice
            else: # invalid config proposed by GS
                if not self._use_rs:
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, {}, error=True) # tell GS there is an error
                self._use_rs = False
                config = self._search_thread_pool[backup].suggest(trial_id)
                skip = self._should_skip(backup, trial_id, config)
                if skip: 
                    return None
                self._trial_proposed_by[trial_id] = backup
                choice = backup
            # if choice: self._pending.add(choice) # local search thread pending
            if not choice:
                if self._ls._resource: 
                # TODO: add resource to config proposed by GS, min or median?
                    config[self._ls.prune_attr] = self._ls.min_resource
            self._result[self._ls.config_signature(config)] = {}
        else: # use init config
            init_config = self._points_to_evaluate.pop(
                0) if self._points_to_evaluate else self._ls.init_config
            config = self._ls.complete_config(init_config,
             self._admissible_min, self._admissible_max)
                # logger.info(f"reset config to {config}")
            config_signature = self._ls.config_signature(config)
            result = self._result.get(config_signature)
            if result: # tried before
                # self.on_trial_complete(trial_id, result)
                return None
            elif result is None: # not tried before
                self._result[config_signature] = {}
            else: return None # running but no result yet
            self._init_used = True
        # logger.info(f"config={config}")
        self._config_results[trial_id] = config
        print(f"config={config}")
        clean_config = enforce_conditions_on_config(config)
        print(f"clean_config={clean_config}")
        return clean_config

    def _should_skip(self, choice, trial_id, config) -> bool:
        ''' if config is None or config's result is known or above mem threshold
            return True; o.w. return False
        '''
        if config is None: return True
        config_signature = self._ls.config_signature(config)
        exists = config_signature in self._result
        # check mem constraint
        if not exists and self._mem_threshold and self._mem_size(
            config)>self._mem_threshold:
            self._result[config_signature] = {
                self._metric:np.inf*self._ls.metric_op, 'time_total_s':1}
            exists = True
        if exists:
            if not self._use_rs:
                result = self._result.get(config_signature)
                if result:
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, result, error=False)
                    if choice:
                        # local search thread
                        self._clean(choice)
                else:
                    # tell the thread there is an error
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, {}, error=True) 
            return True
        return False

    def _select_thread(self) -> Tuple:
        ''' thread selector; use can_suggest to check LS availability
        '''
        # update priority
        min_eci = self._deadline - time.time()
        if min_eci <= 0: return -1, -1
        max_speed = 0
        for thread in self._search_thread_pool.values():            
            if thread.speed > max_speed: max_speed = thread.speed
        for thread in self._search_thread_pool.values():            
            thread.update_eci(self._metric_target, max_speed)
            if thread.eci < min_eci: min_eci = thread.eci
        for thread in self._search_thread_pool.values():
            thread.update_priority(min_eci)

        top_thread_id = backup_thread_id = 0
        priority1 = priority2 = self._search_thread_pool[0].priority
        # logger.debug(f"priority of thread 0={priority1}")
        for thread_id, thread in self._search_thread_pool.items():
            # if thread_id:
            #     logger.debug(
            #         f"priority of thread {thread_id}={thread.priority}")
            #     logger.debug(
            #         f"thread {thread_id}.can_suggest={thread.can_suggest}")
            if thread_id and thread.can_suggest:
                priority = thread.priority
                if priority > priority1: 
                    priority1 = priority
                    top_thread_id = thread_id
                if priority > priority2 or backup_thread_id == 0:
                    priority2 = priority
                    backup_thread_id = thread_id
        return top_thread_id, backup_thread_id

    def _valid(self, config: Dict) -> bool:
        ''' config validator
        '''
        for key in self._admissible_min:
            if key in config:
                value = config[key]
                # logger.info(
                #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                if value<self._admissible_min[
                    key] or value>self._admissible_max[key]:
                    return False
        return True


class CFO(BlendSearch):
    ''' class for CFO algorithm
    '''

    __name__ = 'CFO'

    def suggest(self, trial_id: str) -> Optional[Dict]:
        # Number of threads is 1 or 2. Thread 0 is a vacuous thread
        assert len(self._search_thread_pool)<3, len(self._search_thread_pool)
        if len(self._search_thread_pool) < 2:
            # When a local converges, the number of threads is 1
            # Need to restart
            self._init_used = False
        return super().suggest(trial_id)

    def _select_thread(self) -> Tuple:
        for key in self._search_thread_pool:
            if key: return key, key

    def _create_condition(self, result: Dict) -> bool:
        ''' create thread condition
        '''
        return len(self._search_thread_pool) < 2