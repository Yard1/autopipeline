import numpy as np
import pandas as pd

import gc
import os
import tempfile

from ray import tune

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.model_selection._search import ParameterGrid

from .utils import ray_context, split_list_into_chunks, get_all_tunable_params
from ..utils import call_component_if_needed
from ...components import Component, ComponentConfig
from ...components.flow.pipeline import TopPipeline
from ...problems import ProblemType
from ...search.stage import AutoMLStage
from ...utils.string import removesuffix
from ...utils.exceptions import validate_type

import logging

logger = logging.getLogger(__name__)


def remove_component_suffix(key: str):
    split_key = [s for s in key.split("__") if s[-1] != Component._automl_id_sign]
    return "__".join(split_key)


class Tuner:
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
    ) -> None:
        self.problem_type = problem_type
        self.pipeline_blueprint = pipeline_blueprint
        self.cv = cv
        self.random_state = random_state

    def _get_single_default_hyperparams(self, components):
        hyperparams = {}
        valid_keys = set()
        for k, v in components.items():
            if self._is_component_valid(v, components["Estimator"]):
                valid_keys.add(k)
                for k2, v2 in v.get_tuning_grid().items():
                    name = v.get_hyperparameter_key_suffix(k, k2)
                    hyperparams[name] = v2.default
        return {**{k:v for k,v in components.items() if k in valid_keys}, **hyperparams}

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
            k: v.values for k, v in pipeline_blueprint.get_all_distributions().items()
        }
        default_grid_list = [
            self._get_single_default_hyperparams(components)
            for components in ParameterGrid(default_grid)
        ]
        default_grid_list = [
            components
            for components in default_grid_list
            if self._are_components_valid(components)
        ]
        return default_grid_list

    def _pre_search(self, X, y, groups=None):
        self.X_ = X
        self.y_ = y
        self.groups_ = groups
        self.default_grid = self._get_default_components(self.pipeline_blueprint)
        preset_configurations = [
            config
            for config in self.pipeline_blueprint.preset_configurations
            if config not in self.default_grid
        ]
        self.default_grid += preset_configurations

        self._set_up_early_stopping(X, y, groups=groups)

    def _set_up_early_stopping(self, X, y, groups=None):
        pass

    def _run_search(self):
        raise NotImplementedError()


class RayTuneTuner(Tuner):
    def __init__(
        self,
        problem_type: ProblemType,
        pipeline_blueprint,
        cv,
        random_state,
        num_samples: int = 50,
        cache=False,
        **tune_kwargs,
    ) -> None:
        self.cache = cache
        self._set_cache()
        self.tune_kwargs = tune_kwargs
        self._tune_kwargs = {
            "run_or_experiment": self._trial_with_cv,
            "search_alg": None,
            "scheduler": None,
            "num_samples": num_samples,
            "verbose": 2,
            "reuse_actors": True,
            "fail_fast": True,
        }
        super().__init__(
            problem_type=problem_type,
            pipeline_blueprint=pipeline_blueprint,
            cv=cv,
            random_state=random_state,
        )

    def _set_cache(self):
        validate_type(self.cache, "cache", (str, bool))
        if not self.cache:
            self._cache = None
        else:
            self._cache = self.cache

        if self._cache:
            logger.info(f"Cache dir set as '{self._cache}'")

    def _treat_config(self, config):
        config = {k: self._component_strings_.get(v, v) for k, v in config.items()}
        return {
            remove_component_suffix(k): call_component_if_needed(
                v, random_state=self.random_state
            )
            for k, v in config.items()
        }

    def _trial_with_cv(self, config, checkpoint_dir=None):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config_called = self._treat_config(config)

        estimator.set_params(**config_called)
        memory = tempfile.gettempdir() if self._cache is True else self._cache
        memory = memory if not memory == os.getcwd() else ".."
        estimator.set_params(memory=memory)

        for idx, fraction in enumerate(self.early_stopping_fractions_):
            if len(self.early_stopping_fractions_) > 1 and fraction < 1.0:
                subsample_cv = _SubsampleMetaSplitter(
                    base_cv=self.cv,
                    fraction=fraction,
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
            gc.collect()
            tune.report(
                done=idx + 1 >= len(self.early_stopping_fractions_),
                mean_test_score=np.mean(scores["test_score"]),
                dataset_fraction_=fraction,
            )

    def _pre_search(self, X, y, groups=None):
        super()._pre_search(X, y, groups=groups)
        self._tune_kwargs["num_samples"] += len(self.default_grid)
        _, self._component_strings_ = get_all_tunable_params(self.pipeline_blueprint)
        for conf in self.default_grid:
            for k, v in conf.items():
                if str(v) in self._component_strings_:
                    conf[k] = str(v)

    def _run_search(self):
        tune_kwargs = {**self._tune_kwargs, **self.tune_kwargs}
        gc.collect()
        with ray_context(
            global_checkpoint_s=tune_kwargs.pop("TUNE_GLOBAL_CHECKPOINT_S", 10)
        ):
            self.analysis_ = tune.run(**tune_kwargs)
