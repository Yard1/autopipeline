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
import time
import pickle
from ray.tune.suggest import Searcher
from ray.tune.suggest.optuna import OptunaSearch
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

import optuna as ot
from optuna.trial._state import TrialState
from optuna.distributions import (
    UniformDistribution,
    IntUniformDistribution,
    IntLogUniformDistribution,
    LogUniformDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
)

conditional_space = {"cond_param1": ("cond_param2", [False])}


def enforce_conditions_on_config(config, prefix=""):
    config = config.copy()
    for independent_name, v in conditional_space.items():
        dependent_name, required_values = v
        if f"{prefix}{independent_name}" not in config or (
            config[f"{prefix}{independent_name}"] not in required_values
            and f"{prefix}{dependent_name}" in config
        ):
            config.pop(f"{prefix}{dependent_name}", None)
    return config


def convert_optuna_params_to_distributions(params):
    distributions = {}
    for k, v in params.items():
        fn, args, kwargs = v
        args = args[1:] if len(args) > 0 else args
        kwargs = kwargs.copy()
        kwargs.pop("name", None)
        if fn == "suggest_loguniform":
            distributions[k] = LogUniformDistribution(*args, **kwargs)
        elif fn == "suggest_discrete_uniform":
            distributions[k] = DiscreteUniformDistribution(*args, **kwargs)
        elif fn == "suggest_uniform":
            distributions[k] = UniformDistribution(*args, **kwargs)
        elif fn == "suggest_int":
            if kwargs.pop("log", False) or args[-1] is True:
                if args[-1] is True:
                    args = args[:-1]
                distributions[k] = IntLogUniformDistribution(*args, **kwargs)
            else:
                distributions[k] = IntUniformDistribution(*args, **kwargs)
        elif fn == "suggest_categorical":
            distributions[k] = CategoricalDistribution(*args, **kwargs)
        else:
            raise ValueError(f"Unknown distribution suggester {fn}")
    return distributions


class OptunaConditional(OptunaSearch):
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
            params = self._get_params(ot_trial)
        return unflatten_dict(params)

    def add_evaluated_trial(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
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

        if trial_id in self._ot_trials:
            return False

        config = {k[7:]: v for k, v in result.items() if k.startswith("config/")}
        space_dict = {
            self._get_name(args, kwargs): (fn, args, kwargs)
            for fn, args, kwargs in self._space
        }
        distributions = {k: v for k, v in space_dict.items() if k in config}
        distributions = convert_optuna_params_to_distributions(distributions)

        trial = ot.trial.create_trial(
            state=TrialState.COMPLETE,
            value=result.get(self.metric, None),
            params=config,
            distributions=distributions,
        )

        len_studies = len(self._ot_study.trials)
        self._ot_study.add_trial(trial)
        assert len_studies < len(self._ot_study.trials)

        return True

    def _get_name(self, args, kwargs):
        return args[0] if len(args) > 0 else kwargs["name"]

    def _get_optuna_trial_value(self, ot_trial, tpl):
        fn, args, kwargs = tpl
        return getattr(ot_trial, fn)(*args, **kwargs)

    def _get_params(self, ot_trial):
        params_checked = set()
        params = {}
        space_dict = {
            self._get_name(args, kwargs): (fn, args, kwargs)
            for fn, args, kwargs in self._space
        }
        for key, condition in conditional_space.items():
            value = self._get_optuna_trial_value(ot_trial, space_dict[key])
            params[key] = value
            params_checked.add(key)
            params_checked.add(condition[0])
            if value in condition[1]:
                params[condition[0]] = self._get_optuna_trial_value(
                    ot_trial, space_dict[condition[0]]
                )

        for key, tpl in space_dict.items():
            if key in params_checked:
                continue
            value = self._get_optuna_trial_value(ot_trial, space_dict[key])
            params[key] = value
        return params


GlobalSearch = OptunaConditional


class SharingSearchThread(SearchThread):
    """Class of global or local search thread"""

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
                self._search_alg.on_trial_complete(
                    trial_id,
                    enforce_conditions_on_config(result, prefix="config/"),
                    error,
                )
            else:
                self._search_alg.add_evaluated_trial(
                    trial_id, enforce_conditions_on_config(result, prefix="config/")
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
                trial_id, enforce_conditions_on_config(result, prefix="config/")
            )
        if self.cost_attr in result and self.cost_last < result[self.cost_attr]:
            self.cost_last = result[self.cost_attr]
            # self._update_speed()


class ConditionalBlendSearch(BlendSearch):
    """class for BlendSearch algorithm"""

    def _init_search(self):
        """initialize the search"""
        super()._init_search()
        self._suggested_configs = {}
        self._search_thread_pool = {
            # id: int -> thread: SearchThread
            0: SharingSearchThread(self._ls.mode, self._gs)
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
        ) = save_object

    def restore_from_dir(self, checkpoint_dir: str):
        super.restore_from_dir(checkpoint_dir)

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
                self._search_thread_pool[self._thread_count] = SharingSearchThread(
                    self._ls.mode,
                    self._ls.create(
                        config, result[self._metric], cost=result["time_total_s"]
                    ),
                )
                thread_id = self._thread_count
                self._thread_count += 1

        # cleaner
        # logger.info(f"thread {thread_id} in search thread pool="
        #     f"{thread_id in self._search_thread_pool}")
        if thread_id and thread_id in self._search_thread_pool:
            # local search thread
            self._clean(thread_id)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """choose thread, suggest a valid config"""
        if self._init_used and not self._points_to_evaluate:
            choice, backup = self._select_thread()
            # logger.debug(f"choice={choice}, backup={backup}")
            if choice < 0:
                return None  # timeout
            self._use_rs = False
            config = self._search_thread_pool[choice].suggest(trial_id)
            if len(config) != len(self._ls.space):
                try:
                    config = {
                        **self._search_thread_pool[backup]._search_alg.best_config,
                        **config,
                    }
                    assert len(config) == len(self._ls.space)
                except:
                    for _, generated in generate_variants({"config": self._ls.space}):
                        config = {**generated["config"], **config}
                        break
            skip = self._should_skip(choice, trial_id, config)
            if skip:
                if choice:
                    # logger.info(f"skipping choice={choice}, config={config}")
                    return None
                # use rs
                self._use_rs = True
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
                self._trial_proposed_by[trial_id] = choice
            else:  # invalid config proposed by GS
                if not self._use_rs:
                    self._search_thread_pool[choice].on_trial_complete(
                        trial_id, {}, error=True
                    )  # tell GS there is an error
                self._use_rs = False
                config = self._search_thread_pool[backup].suggest(trial_id)
                if len(config) != len(self._ls.space):
                    try:
                        config = {
                            **self._search_thread_pool[choice]._search_alg.best_config,
                            **config,
                        }
                        assert len(config) == len(self._ls.space)
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
        assert len(config) == len(self._ls.space)
        self._suggested_configs[trial_id] = config
        clean_config = enforce_conditions_on_config(config)
        return clean_config

    def _has_config_been_already_tried(self, config) -> bool:
        config_signature = self._ls.config_signature(config)
        if not conditional_space:
            return (
                (self._result.get(config_signature, None) in self._result),
                config_signature,
                None,
            )
        enforced_config_signature = self._ls.config_signature(
            enforce_conditions_on_config(config)
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
                "time_total_s": 1,
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