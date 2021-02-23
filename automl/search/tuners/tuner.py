import numpy as np
import pandas as pd

import gc
import platform

from ray import tune

from sklearn.model_selection import cross_validate
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter
from sklearn.model_selection._search import ParameterGrid

from .utils import ray_context, split_list_into_chunks
from ..utils import call_component_if_needed
from ...components import Component, ComponentConfig
from ...components.flow.pipeline import TopPipeline
from ...problems import ProblemType
from ...search.stage import AutoMLStage
from ...utils.string import removesuffix
from ...utils.exceptions import validate_type


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
        for k, v in components.items():
            for k2, v2 in v.get_tuning_grid().items():
                name = v.get_hyperparameter_key_suffix(k, k2)
                hyperparams[name] = v2.default
        return {**components, **hyperparams}

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
        elif self.cache is True:
            self._cache = (
                "/dev/shm" if platform.system() == "Linux" else ".."
            )
        else:
            self._cache = self.cache

    def _trial_with_cv(self, config, checkpoint_dir=None):
        estimator = self.pipeline_blueprint(random_state=self.random_state)

        config_called = {
            remove_component_suffix(k): call_component_if_needed(
                v, random_state=self.random_state
            )
            for k, v in config.items()
        }

        estimator.set_params(**config_called)
        estimator.memory = self._cache

        for idx, fraction in enumerate(self.early_stopping_fractions_):
            if len(self.early_stopping_fractions_) > 1:
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

            tune.report(
                done=idx + 1 >= len(self.early_stopping_fractions_),
                mean_test_score=np.mean(scores["test_score"]),
                dataset_fraction=fraction,
            )

    def _pre_search(self, X, y, groups=None):
        super()._pre_search(X, y, groups=groups)
        self._tune_kwargs["num_samples"] += len(self.default_grid)

    def _run_search(self):
        tune_kwargs = {**self._tune_kwargs, **self.tune_kwargs}
        gc.collect()
        with ray_context(
            global_checkpoint_s=tune_kwargs.pop("TUNE_GLOBAL_CHECKPOINT_S", 10)
        ):
            self.analysis_ = tune.run(**tune_kwargs)
