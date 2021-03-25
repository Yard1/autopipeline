from automl.utils.display import IPythonDisplay
from automl.utils.tune_callbacks import BestPlotCallback
import plotly.graph_objs as go


from time import sleep
import numpy as np
import pandas as pd

import gc
from abc import ABC

import ray
from ray import tune

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.model_selection._search import ParameterGrid

from .trainable import SklearnTrainable, RayStore
from .utils import get_all_tunable_params
from ..utils import ray_context
from ...components import Component, ComponentConfig
from ...components.flow.pipeline import TopPipeline
from ...components.transformers.passthrough import Passthrough
from ...problems import ProblemType
from ...search.stage import AutoMLStage
from ...utils.string import removesuffix
from ...utils.exceptions import validate_type
from ...utils.memory import dynamic_memory_factory

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
        target_metric=None,
        scoring=None,
    ) -> None:
        self.problem_type = problem_type
        self.pipeline_blueprint = pipeline_blueprint
        self.cv = cv
        self.random_state = random_state
        self.use_extended = use_extended
        self.num_samples = num_samples
        self.time_budget_s = time_budget_s
        self.target_metric = target_metric
        self.scoring = scoring
        if target_metric:
            assert target_metric in scoring

    def _get_single_default_hyperparams(self, components, grid):
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

    def _is_component_valid(self, component, estimator):
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
            k: v.values
            for k, v in pipeline_blueprint.get_all_distributions(
                use_extended=self.use_extended
            ).items()
        }
        default_grid_list = [
            self._get_single_default_hyperparams(components, default_grid)
            for components in ParameterGrid(default_grid)
        ]
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

        default_grid_list_dict = []
        default_grid_list_no_dups = []
        for config in self.default_grid_:
            str_config = {
                k: str(v) for k, v in config.items() if not isinstance(v, Passthrough)
            }
            if str_config not in default_grid_list_dict:
                default_grid_list_dict.append(str_config)
                default_grid_list_no_dups.append(config)
        self.default_grid_ = default_grid_list_no_dups

        self._set_up_early_stopping(X, y, groups=groups)

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
        target_metric=None,
        scoring=None,
        cache=False,
        **tune_kwargs,
    ) -> None:
        self.cache = cache
        self._set_cache()
        self.tune_kwargs = tune_kwargs
        self.num_samples = num_samples
        self._tune_kwargs = {
            "run_or_experiment": None,
            "search_alg": None,
            "scheduler": None,
            "num_samples": num_samples,
            "time_budget_s": time_budget_s,
            "verbose": 2,
            "reuse_actors": True,
            "fail_fast": True,  # TODO change to False when ready
            "resources_per_trial": {"cpu": 1},
            "run_or_experiment": SklearnTrainable,
            "stop": {"training_iteration": 1},
            # "max_failures": 2
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

    def _pre_search(self, X, y, X_test=None, y_test=None, groups=None):
        super()._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)
        if self._cache:
            np.random.shuffle(self.default_grid_)
        _, self._component_strings_ = get_all_tunable_params(
            self.pipeline_blueprint, use_extended=self.use_extended
        )
        for conf in self.default_grid_:
            for k, v in conf.items():
                if str(v) in self._component_strings_:
                    conf[k] = str(v)

    def _run_search(self):
        tune_kwargs = {**self._tune_kwargs, **self.tune_kwargs}
        # TODO make this better
        display = IPythonDisplay("best_plot_display")
        widget = go.FigureWidget()
        widget.add_scatter(mode='lines+markers', name="Best score")
        widget.add_scatter(mode='lines', name="Mean score")
        display.display(widget)
        plot_callback = BestPlotCallback(widget=widget, metric="mean_validation_score")
        tune_kwargs["num_samples"] = self.total_num_samples
        tune_kwargs["callbacks"] = tune_kwargs.get("callbacks", [])
        tune_kwargs["callbacks"].append(plot_callback)
        params = {
            "X_": self.X_,
            "y_": self.y_,
            "X_test_": self.X_test_,
            "y_test_": self.y_test_,
            "pipeline_blueprint": self.pipeline_blueprint,
            "_component_strings_": self._component_strings_,
            "groups_": self.groups_,
            "fit_params": None,
            "scoring": self.scoring,
            "metric_name": self.target_metric,
            "cv": self.cv,
            "n_jobs": None,
            "random_state": self.random_state,
            "prune_attr": self._searcher_kwargs.get("prune_attr", None),
            "cache": self._cache,
        }
        tune_kwargs["run_or_experiment"] = tune.with_parameters(
            tune_kwargs["run_or_experiment"], **params
        )
        gc.collect()
        with ray_context(
            global_checkpoint_s=tune_kwargs.pop("TUNE_GLOBAL_CHECKPOINT_S", 10)
        ):
            object_store = RayStore.options(
                name="object_store"
            ).remote()

            self.analysis_ = tune.run(**tune_kwargs)

            gc.collect()

            self.fold_predictions_ = {}
            self.fitted_estimators_ = {}

            logger.debug("getting estimators from ray object store")
            trial_ids = ray.get(object_store.get_all_keys.remote("fitted_estimators"))
            fitted_estimators = [
                RayStore.decompress(
                    ray.get(object_store.get.remote(key, "fitted_estimators", pop=True, decompress=False))
                )
                for key in trial_ids
            ]
            self.fitted_estimators_ = dict(zip(trial_ids, fitted_estimators))

            logger.debug("getting fold predictions from ray object store")
            trial_ids = ray.get(object_store.get_all_keys.remote("fold_predictions"))
            fold_predictions = [
                RayStore.decompress(
                    ray.get(
                        object_store.get.remote(
                            key, "fold_predictions", pop=True, decompress=False
                        )
                    )
                )
                for key in trial_ids
            ]
            fold_predictions = dict(zip(trial_ids, fold_predictions))
            for key in self.analysis_.results.keys():
                self.fold_predictions_[key] = fold_predictions.get(key, {})

            del object_store

            gc.collect()

    def _search(self, X, y, X_test=None, y_test=None, groups=None):
        self._pre_search(X, y, X_test=X_test, y_test=y_test, groups=groups)

        self._run_search()

        return self

    def fit(self, X, y, X_test=None, y_test=None, groups=None):
        return self._search(X, y, X_test=X_test, y_test=y_test, groups=groups)