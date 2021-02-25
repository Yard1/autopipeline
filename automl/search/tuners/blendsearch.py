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
import tempfile
from ray.tune.sample import Categorical

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter

from ray import tune
from ray.tune.suggest import Searcher
from ray.tune.suggest.variant_generator import generate_variants
from flaml.searcher.search_thread import SearchThread
from flaml.searcher.blendsearch import BlendSearch
from flaml.searcher.flow2 import FLOW2

LocalSearch = FLOW2

import logging

logger = logging.getLogger(__name__)

from ray.tune.suggest import Searcher

from ray.tune.suggest.suggestion import (
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
)
from ray.tune.utils.util import unflatten_dict

from .OptunaTPETuner import ConditionalOptunaSearch
from ..distributions import get_tune_distributions
from .utils import get_conditions, enforce_conditions_on_config, get_all_tunable_params
from .tuner import RayTuneTuner, remove_component_suffix
from ..utils import call_component_if_needed
from ...problems import ProblemType


GlobalSearch = ConditionalOptunaSearch


class SharingSearchThread(SearchThread):
    """Class of global or local search thread"""

    def __init__(
        self, mode: str = "min", search_alg: Optional[Searcher] = None, cost_attr=None
    ):
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
        if cost_attr:
            self.cost_attr = cost_attr

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        """update the statistics of the thread"""
        if not self._search_alg:
            return
        if not hasattr(self._search_alg, "_ot_trials"):
            # optuna doesn't handle error
            self._search_alg.on_trial_complete(trial_id, result, error)
        elif not error:
            if trial_id in self._search_alg._ot_trials:
                print(f"on_trial_complete trial_id {trial_id}")
                if np.around(result["dataset_fraction"], 1) >= 1.0:
                    print("adding trial to optuna")
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
                    print("reporting trial to optuna")
                    self._search_alg.on_trial_result(
                        trial_id,
                        enforce_conditions_on_config(
                            result,
                            self._search_alg._conditional_space,
                            prefix="config/",
                            keys_to_keep=self._search_alg._space,
                        ),
                    )
            elif np.around(result["dataset_fraction"], 1) >= 1.0:
                print("add_evaluated_trial")
                print(result)
                self._search_alg.add_evaluated_trial(
                    trial_id,
                    enforce_conditions_on_config(
                        result,
                        self._search_alg._conditional_space,
                        prefix="config/",
                        keys_to_keep=self._search_alg._space,
                    ),
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
                keep_const_values=False,
            )
        else:
            self._gs = None

        space, _ = get_all_tunable_params(space, to_str=True)
        space = get_tune_distributions(space)
        self._const_values = {
            k: v.categories[0]
            for k, v in space.items()
            if isinstance(v, Categorical) and len(v.categories) == 1
        }
        space = {k: v for k, v in space.items() if k not in self._const_values}

        if points_to_evaluate:
            points_to_evaluate = [
                {k: v for k, v in point.items() if k in space}
                for point in points_to_evaluate
            ]
            init_config = points_to_evaluate[0]
        else:
            init_config = {}
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
        self._resources_per_trial = resources_per_trial
        self._mem_size = mem_size
        self._mem_threshold = (
            resources_per_trial.get("mem") if resources_per_trial else None
        )
        self._init_search()

    @property
    def keys_to_keep(self):
        return set.union(
            set(self._ls.space),
            set(self._ls.prune_attr) if self._ls.prune_attr else set(),
        )

    def _init_search(self):
        """initialize the search"""
        super()._init_search()
        self._suggested_configs = {}
        self._search_thread_pool = {
            # id: int -> thread: SearchThread
            0: SharingSearchThread(
                self._ls.mode, self._gs, cost_attr=self._ls.prune_attr
            )
        }

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
            self._admissible_min,
            self._admissible_max,
            self._result,
            self._deadline,
            self._suggested_configs,
            self._conditional_space,
            self._time_attr,
            self._const_values,
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
            self._admissible_min,
            self._admissible_max,
            self._result,
            self._deadline,
            self._suggested_configs,
            self._conditional_space,
            self._time_attr,
            self._const_values,
        ) = save_object

    def restore_from_dir(self, checkpoint_dir: str):
        super.restore_from_dir(checkpoint_dir)

    def on_trial_result(self, trial_id: str, result: Dict):
        if trial_id not in self._trial_proposed_by:
            return
        thread_id = self._trial_proposed_by[trial_id]
        if not thread_id in self._search_thread_pool:
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
            del self._trial_proposed_by[trial_id]
            # if not thread_id: logger.info(f"result {result}")
        if result:
            config = self._suggested_configs[trial_id]
            assert len(config) == len(self._ls.space) + int(bool(self._ls.prune_attr))
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
            if thread_id:  # from local search
                # update admissible region
                normalized_config = self._ls.normalize(config)
                for key in self._admissible_min:
                    value = normalized_config[key]
                    if value > self._admissible_max[key]:
                        self._admissible_max[key] = value
                    elif value < self._admissible_min[key]:
                        self._admissible_min[key] = value
                if self._global_search_thread:
                    self._global_search_thread.on_trial_complete(
                        trial_id, result, error
                    )
            elif self._create_condition(result):
                # thread creator
                assert len(config) == len(self._ls.space) + int(
                    bool(self._ls.prune_attr)
                )
                self._search_thread_pool[self._thread_count] = SharingSearchThread(
                    self._ls.mode,
                    self._ls.create(
                        config, result[self._metric], cost=result[self._time_attr]
                    ),
                    cost_attr=self._ls.prune_attr,
                )
                thread_id = self._thread_count
                self._thread_count += 1

        # cleaner
        # logger.info(f"thread {thread_id} in search thread pool="
        #     f"{thread_id in self._search_thread_pool}")
        if thread_id and thread_id in self._search_thread_pool:
            # local search thread
            self._clean(thread_id)

    def _valid(self, config: Dict) -> bool:
        """config validator"""
        print("_valid")
        print(config)
        print(self._admissible_min)
        try:
            for key in self._admissible_min:
                if key in config:
                    value = config[key]
                    # logger.info(
                    #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                    if (
                        value < self._admissible_min[key]
                        or value > self._admissible_max[key]
                    ):
                        return False
        except TypeError:
            normalized_config = self._ls.normalize(config)
            for key in self._admissible_min:
                if key in normalized_config:
                    value = normalized_config[key]
                    # logger.info(
                    #     f"{key},{value},{self._admissible_min[key]},{self._admissible_max[key]}")
                    if (
                        value < self._admissible_min[key]
                        or value > self._admissible_max[key]
                    ):
                        return False
        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """choose thread, suggest a valid config"""
        print("suggest")
        if self._init_used and not self._points_to_evaluate:
            choice, backup = self._select_thread()
            # logger.debug(f"choice={choice}, backup={backup}")
            if choice < 0:
                return None  # timeout
            self._use_rs = False
            config = self._search_thread_pool[choice].suggest(trial_id)
            config = {k: v for k, v in config.items() if k not in self.keys_to_keep}
            if not config or len(config) != len(self._ls.space):
                try:
                    config = {
                        **self._search_thread_pool[backup]._search_alg.best_config,
                        **config,
                    }
                    print(config)
                    assert len(config) == len(self._ls.space) or len(config) == len(
                        self._ls.space
                    ) + int(bool(self._ls.prune_attr))
                except:
                    print("assertion failed")
                    for _, generated in generate_variants({"config": self._ls.space}):
                        config = {**generated["config"], **config}
                        break
                    print(config)
            skip = self._should_skip(choice, trial_id, config)
            if skip:
                print("skipping")
                if choice:
                    # logger.info(f"skipping choice={choice}, config={config}")
                    return None
                # use rs
                self._use_rs = True
                print("using rs")
                for _, generated in generate_variants({"config": self._ls.space}):
                    config = generated["config"]
                    break
                # logger.debug(f"random config {config}")
                skip = self._should_skip(choice, trial_id, config)
                if skip:
                    return None
            # if not choice: logger.info(config)
            if choice or backup == choice or self._valid(config):
                # LS or valid or no backup choice
                print("LS or valid or no backup choice")
                self._trial_proposed_by[trial_id] = choice
            else:  # invalid config proposed by GS
                print("invalid config proposed by GS")
                if not self._use_rs:
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, {}, error=True
                    )  # tell GS there is an error
                self._use_rs = False
                config = self._search_thread_pool[backup].suggest(trial_id)
                if not config or len(config) != len(self._ls.space):
                    try:
                        config = {
                            **self._search_thread_pool[choice]._search_alg.best_config,
                            **config,
                        }
                        assert len(config) == len(self._ls.space) or len(config) == len(
                            self._ls.space
                        ) + int(bool(self._ls.prune_attr))
                    except:
                        for _, generated in generate_variants(
                            {"config": self._ls.space}
                        ):
                            config = {**generated["config"], **config}
                            break
                skip = self._should_skip(backup, trial_id, config)
                if skip:
                    return None
                self._trial_proposed_by[trial_id] = backup
                choice = backup
            # if choice: self._pending.add(choice) # local search thread pending
            if not choice:
                print("choice is none")
                if self._ls._resource:
                    # TODO: add resource to config proposed by GS, min or median?
                    config[self._ls.prune_attr] = self._ls.min_resource
            (
                result,
                config_signature,
                enforced_config_signature,
            ) = self._has_config_been_already_tried(config)
            self._result[config_signature] = {}
            if enforced_config_signature:
                self._result[enforced_config_signature] = {}
        else:  # use init config
            print("using init config")
            init_config = (
                self._points_to_evaluate.pop(0)
                if self._points_to_evaluate
                else self._ls.init_config
            )
            config = self._ls.complete_config(
                init_config, self._admissible_min, self._admissible_max
            )
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
        # logger.info(f"config={config}")
        print(list(config.keys()))
        print(list(self._ls.space.keys()))
        assert len(config) == len(self._ls.space) + int(bool(self._ls.prune_attr))
        self._suggested_configs[trial_id] = config
        clean_config = enforce_conditions_on_config(config, self._conditional_space)
        clean_config = {**self._const_values, **clean_config}
        print("CONFIG")
        print(clean_config)
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
            enforce_conditions_on_config(config, self._conditional_space)
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
            not exists
            and self._mem_threshold
            and self._mem_size(config) > self._mem_threshold
        ):
            self._result[config_signature] = {
                self._metric: np.inf * self._ls.metric_op,
                self._time_attr: 1,
            }
            exists = True
        if exists:
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
                else:
                    # tell the thread there is an error
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, {}, error=True
                    )
            return True
        return False


class BlendSearchTuner(RayTuneTuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        num_samples: int = 20,
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
        if self.early_stopping:
            min_dist = self.cv.get_n_splits(self.X_, self.y_, self.groups_) * 2
            if self.problem_type.is_classification():
                min_dist *= len(self.y_.cat.categories)
            min_dist /= self.X_.shape[0]

            step = 4
            self.early_stopping_fractions_ = [min_dist]
            while self.early_stopping_fractions_[-1] < 1.0:
                self.early_stopping_fractions_.append(
                    self.early_stopping_fractions_[-1] * step
                )
            self.early_stopping_fractions_[-1] = 1.0
            assert (
                self.early_stopping_fractions_[0] < self.early_stopping_fractions_[1]
            ), f"Could not generate correct fractions for the given number of splits. {self.early_stopping_fractions_}"
            assert (
                self.early_stopping_fractions_[-1] > self.early_stopping_fractions_[-2]
            ), f"Could not generate correct fractions for the given number of splits. {self.early_stopping_fractions_}"
            self._searcher_kwargs["prune_attr"] = "dataset_fraction"
            self._searcher_kwargs["min_resource"] = self.early_stopping_fractions_[0]
            self._searcher_kwargs["max_resource"] = self.early_stopping_fractions_[-1]
            self._searcher_kwargs["reduction_factor"] = step
        else:
            self.early_stopping_fractions_ = [1]
        print(self.early_stopping_fractions_)

    def _pre_search(self, X, y, groups=None):
        super()._pre_search(X, y, groups=groups)
        _, self._component_strings_ = get_all_tunable_params(self.pipeline_blueprint)
        for conf in self.default_grid:
            for k, v in conf.items():
                if str(v) in self._component_strings_:
                    conf[k] = str(v)
        self._tune_kwargs["search_alg"] = ConditionalBlendSearch(
            space=self.pipeline_blueprint,
            metric="mean_test_score",
            mode="max",
            points_to_evaluate=self.default_grid,
            seed=self.random_state,
            **self._searcher_kwargs,
        )

    def _treat_config(self, config):
        config = {k: self._component_strings_.get(v, v) for k, v in config.items()}
        return super()._treat_config(config)

    def _trial_with_cv(self, config, checkpoint_dir=None):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        prune_attr = self._searcher_kwargs.get("prune_attr")

        config_called = self._treat_config(config)
        config_called.pop(prune_attr, None)

        if prune_attr:
            prune_attr = config.get(prune_attr)

        print(f"prune_attr: {prune_attr}")

        estimator.set_params(**config_called)
        memory = tempfile.gettempdir() if self._cache is True else self._cache
        memory = memory if not memory == os.getcwd() else ".."
        estimator.set_params(memory=memory)

        if prune_attr:
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
            # fit_params=self.fit_params,
            # groups=self.groups,
            # return_train_score=self.return_train_score,
            # scoring=self.scoring,
        )

        if prune_attr:
            tune.report(
                done=True,
                mean_test_score=np.mean(scores["test_score"]),
                dataset_fraction=prune_attr,
            )
        else:
            tune.report(
                done=True,
                mean_test_score=np.mean(scores["test_score"]),
            )

    def _search(self, X, y, groups=None):
        self._pre_search(X, y, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, groups=None):
        return self._search(X, y, groups=groups)
