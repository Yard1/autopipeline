from typing import Optional
import plotly.graph_objs as go
from time import sleep
import numpy as np
import pandas as pd
import collections
import gc
from abc import ABC

import ray
import ray.exceptions
from ray import tune

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.model_selection._search import ParameterGrid

from .with_parameters import with_parameters

from .trainable import SklearnTrainable
from .utils import get_all_tunable_params
from ...components import Component, ComponentConfig
from ...components.flow.pipeline import TopPipeline
from ...components.transformers.passthrough import Passthrough
from ...problems import ProblemType
from ...search.stage import AutoMLStage
from ...utils.string import removesuffix
from ...utils.exceptions import validate_type
from ...utils.memory import dynamic_memory_factory
from ...utils.display import IPythonDisplay
from ...utils.tune_callbacks import BestPlotCallback
from ...utils.memory.hashing import hash as xxd_hash

import warnings

import logging

logger = logging.getLogger(__name__)


class Tuner(ABC):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        use_extended: bool = False,
        num_samples: int = -1,
        time_budget_s: int = 600,
        secondary_pipeline_blueprint=None,
        target_metric=None,
        scoring=None,
        display: Optional[IPythonDisplay] = None,
        stacking_level: int = 0,
        previous_stack = None,
    ) -> None:
        self.problem_type = problem_type
        self.pipeline_blueprint = pipeline_blueprint
        self.cv = cv
        self.random_state = random_state
        self.use_extended = use_extended
        self.num_samples = num_samples
        self.time_budget_s = time_budget_s
        self.stacking_level = stacking_level
        self.display = display
        self.previous_stack = previous_stack
        # TODO reenable
        # assert target_metric in scoring

        self.target_metric = target_metric
        self.scoring = scoring
        self.secondary_pipeline_blueprint = secondary_pipeline_blueprint

    def _get_single_default_hyperparams(self, components: dict, grid: list) -> dict:
        hyperparams = {}
        valid_keys = set()
        for k, v in components.items():
            if not self._is_component_valid(v, components["Estimator"]):
                try:
                    v = next(
                        x
                        for x in grid[k]
                        if self._is_component_valid(x, components["Estimator"])
                    )
                except StopIteration:
                    continue
            valid_keys.add(k)
            for k2, v2 in v.get_tuning_grid(use_extended=self.use_extended).items():
                name = v.get_hyperparameter_key_suffix(k, k2)
                hyperparams[name] = v2.default
        return {
            **{k: v for k, v in components.items() if k in valid_keys},
            **hyperparams,
        }

    def _are_components_valid(self, components: dict) -> bool:
        for k, v in components.items():
            if k == "Estimator":
                continue
            if isinstance(v, Component):
                if not v.is_component_valid(
                    config=ComponentConfig(estimator=components["Estimator"]),
                    stage=AutoMLStage.TUNE,
                ):
                    return False
        return True

    def _is_component_valid(self, component, estimator) -> bool:
        if component is estimator:
            return True
        if isinstance(component, Component):
            if not component.is_component_valid(
                config=ComponentConfig(estimator=estimator),
                stage=AutoMLStage.TUNE,
            ):
                return False
        return True

    def _get_default_components(self, pipeline_blueprint) -> dict:
        default_grid = {
            k: [
                component
                for component in v.values
                if component._consider_for_initial_combinations
            ]
            for k, v in pipeline_blueprint.get_all_distributions(
                use_extended=self.use_extended
            ).items()
        }
        default_grid_list = [
            self._get_single_default_hyperparams(components, default_grid)
            for components in ParameterGrid(default_grid)
        ]

        for step_name, classes in pipeline_blueprint.extra_configs.items():
            extra_config_presets = []
            for config in default_grid_list:
                if type(config.get(step_name, None)) in classes:
                    for extra_config in classes[type(config.get(step_name, None))]:
                        extra_config_presets.append(config.copy())
                        for k, v in extra_config.final_parameters.items():
                            name = extra_config.get_hyperparameter_key_suffix(
                                step_name, k
                            )
                            extra_config_presets[-1][name] = (
                                v if v is not None else "!None"
                            )
            default_grid_list.extend(extra_config_presets)

        default_grid_list = [
            components
            for components in default_grid_list
            if self._are_components_valid(components)
        ]

        return default_grid_list

    def _pre_search(self, X, y, X_test=None, y_test=None, groups=None):
        self.X_ = X
        self.y_ = y
        self.X_test_ = X_test
        self.y_test_ = y_test
        self.groups_ = groups
        self.default_grid_ = self._get_default_components(self.pipeline_blueprint)
        preset_configurations = [
            config
            for config in self.pipeline_blueprint.preset_configurations
            if config not in self.default_grid_
        ]
        self.default_grid_ += preset_configurations

        if self.secondary_pipeline_blueprint:
            self.secondary_grid_ = self._get_default_components(
                self.secondary_pipeline_blueprint
            )
        else:
            self.secondary_grid_ = []

        self._remove_duplicates_from_grids()

        self._set_up_early_stopping(X, y, groups=groups)

    def _remove_duplicates_from_grids(self):
        default_grid_list_dict = []
        default_grid_list_no_dups = []
        secondary_grid_list_dict = []
        secondary_grid_list_no_dups = []
        for config in self.default_grid_:
            str_config = {
                k: str(v) for k, v in config.items() if not isinstance(v, Passthrough)
            }
            if str_config not in default_grid_list_dict:
                default_grid_list_dict.append(str_config)
                default_grid_list_no_dups.append(config)

        if self.secondary_grid_:
            for config in self.secondary_grid_:
                str_config = {
                    k: str(v)
                    for k, v in config.items()
                    if not isinstance(v, Passthrough)
                }
                if (
                    str_config not in default_grid_list_dict
                    and str_config not in secondary_grid_list_dict
                ):
                    secondary_grid_list_dict.append(str_config)
                    secondary_grid_list_no_dups.append(config)

        self.default_grid_ = default_grid_list_no_dups
        self.secondary_grid_ = secondary_grid_list_no_dups

    def _set_up_early_stopping(self, X, y, groups=None):
        pass

    def _run_search(self):
        raise NotImplementedError()

    def fit(self, X, y, groups=None):
        raise NotImplementedError()


class RayTuneTuner(Tuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        use_extended: bool = False,
        num_samples: int = -1,
        time_budget_s: int = 600,
        secondary_pipeline_blueprint=None,
        target_metric=None,
        scoring=None,
        cache=False,
        max_concurrent: int = 1,
        trainable_n_jobs: int = 4,
        display: Optional[IPythonDisplay] = None,
        stacking_level: int = 0,
        widget: Optional[go.FigureWidget] = None,
        plot_callback: Optional[BestPlotCallback] = None,
        previous_stack = None,
        **tune_kwargs,
    ) -> None:
        self.cache = cache
        self._set_cache()
        self.tune_kwargs = tune_kwargs
        self.num_samples = num_samples
        self.max_concurrent = max_concurrent
        self.trainable_n_jobs = trainable_n_jobs
        self.widget = widget
        self.plot_callback = plot_callback
        self._tune_kwargs = {
            "run_or_experiment": None,
            "search_alg": None,
            "scheduler": None,
            "num_samples": num_samples,
            "time_budget_s": time_budget_s,
            "verbose": 2,
            "reuse_actors": False,
            "fail_fast": True,  # TODO change to False when ready
            # "resources_per_trial": {"cpu": self.trainable_n_jobs},
            "stop": {"training_iteration": 1},
            "max_failures": 0,
        }
        super().__init__(
            problem_type=problem_type,
            pipeline_blueprint=pipeline_blueprint,
            cv=cv,
            random_state=random_state,
            use_extended=use_extended,
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            scoring=scoring,
            target_metric=target_metric,
            secondary_pipeline_blueprint=secondary_pipeline_blueprint,
            display=display,
            stacking_level=stacking_level,
            previous_stack=previous_stack,
        )

    @property
    def total_num_samples(self):
        if self.num_samples < 0:
            return -1
        return len(self.default_grid_) + self.num_samples

    def _set_cache(self):
        validate_type(self.cache, "cache", (str, bool))
        if not self.cache:
            self._cache = None
        else:
            self._cache = self.cache

        if self._cache:
            logger.info(f"Cache dir set as '{self._cache}'")

    def _shuffle_default_grid(self):
        # default python hash is different on every run
        self.default_grid_.sort(
            key=lambda x: xxd_hash(tuple(k for k in x))
        )
        np.random.default_rng(seed=self.random_state).shuffle(self.default_grid_)

    def _pre_search(self, X, y, X_test=None, y_test=None, groups=None):
        super()._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)
        # this is just to ensure constant order
        self._shuffle_default_grid()
        _, self.component_strings_, self.hyperparameter_names_ = get_all_tunable_params(
            self.secondary_pipeline_blueprint
            if self.secondary_pipeline_blueprint
            else self.pipeline_blueprint,
            use_extended=self.use_extended,
        )
        for conf in self.default_grid_:
            for k, v in conf.items():
                if str(v) in self.component_strings_:
                    conf[k] = str(v)
        for conf in self.secondary_grid_:
            for k, v in conf.items():
                if str(v) in self.component_strings_:
                    conf[k] = str(v)

    def _configure_callbacks(self, tune_kwargs):
        # TODO make this better
        display = self.display or IPythonDisplay("tuner_best_plot_display")
        if not self.widget:
            self.widget_ = go.FigureWidget()
            self.widget_.add_scatter(mode="lines+markers", name="Best validation score")
            self.widget_.add_scatter(mode="lines+markers", name="Best test score")
            self.widget_.add_scatter(mode="lines", name="Mean score")
            self.widget_.add_scatter(mode="markers", name="Validation score")
            display.display(self.widget_)
        else:
            self.widget_ = self.widget
        callbacks = tune_kwargs.get("callbacks", [])
        if self.plot_callback:
            self.plot_callback_ = self.plot_callback
        else:
            self.plot_callback_ = BestPlotCallback(
                widget=self.widget_,
                metric=self.target_metric,
            )  # TODO metric
        tune_kwargs["callbacks"] = callbacks
        tune_kwargs["callbacks"].append(self.plot_callback_)

    def _run_search(self):
        tune_kwargs = {**self._tune_kwargs, **self.tune_kwargs}
        self._configure_callbacks(tune_kwargs)
        tune_kwargs["num_samples"] = self.total_num_samples
        print(f"columns to tune: {self.X_.columns}")
        params = {
            "X_": self.X_,
            "y_": self.y_,
            "X_test_": self.X_test_,
            "y_test_": self.y_test_,
            "pipeline_blueprint": self.pipeline_blueprint,
            "component_strings": self.component_strings_,
            "hyperparameter_names": self.hyperparameter_names_,
            "problem_type": self.problem_type,
            "groups_": self.groups_,
            "fit_params": None,
            "scoring": self.scoring,
            "metric_name": self.target_metric,
            "cv": self.cv,
            "random_state": self.random_state,
            "prune_attr": self._searcher_kwargs.get("prune_attr", None),
            "cache": self._cache,
            "previous_stack": self.previous_stack,
        }
        gc.collect()

        tune_kwargs["run_or_experiment"] = type(
            "SklearnTrainable", (SklearnTrainable,), {"N_JOBS": self.trainable_n_jobs}
        )
        tune_kwargs["run_or_experiment"] = with_parameters(
            tune_kwargs["run_or_experiment"], **params
        )

        self.analysis_ = tune.run(**tune_kwargs)

        gc.collect()

    def _search(self, X, y, X_test=None, y_test=None, groups=None):
        self._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, X_test=None, y_test=None, groups=None):
        return self._search(X, y, X_test=X_test, y_test=y_test, groups=groups)
