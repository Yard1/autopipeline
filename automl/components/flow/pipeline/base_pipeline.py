from typing import List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

from copy import copy, deepcopy
from sklearn.base import clone
from imblearn.pipeline import Pipeline as _ImblearnPipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import is_classifier
from time import time

from ..flow import Flow
from ..utils import (
    recursively_remove_invalid_components,
    get_single_component_from_iterable,
    is_component_valid_iterable,
    get_step_choice_grid,
    recursively_call_tuning_grid_funcs,
)
from ..utils import convert_tuning_grid
from ...transformers import *
from ...estimators import *
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ....search.distributions import CategoricalDistribution
from ....utils.exceptions import validate_type


class BasePipeline(_ImblearnPipeline):
    def set_params(self, **kwargs):
        # ConfigSpace workaround
        kwargs = {k: (None if v == "!None" else v) for k, v in kwargs.items()}
        return super().set_params(**kwargs)

    def _convert_to_df_if_needed(self, X, y=None, fit=False):
        if not hasattr(self, "X_columns_"):
            validate_type(X, "X", pd.DataFrame)
        if fit and isinstance(X, pd.DataFrame):
            self.X_columns_ = X.columns
            self.X_dtypes_ = X.dtypes
        else:
            X = pd.DataFrame(X, columns=self.X_columns_)
            X = X.astype(self.X_dtypes_)
        if y is not None:
            if fit:
                if isinstance(y, pd.Series):
                    self.y_name_ = y.name
                    self.y_dtype_ = y.dtype
                else:
                    self.y_name_ = "target"
                    if is_classifier(self._final_estimator):
                        y = y.astype(int)
                        self.y_dtype_ = "category"
                    else:
                        self.y_dtype_ = np.float32  # TODO make dynamic
                    y = pd.Series(y, name=self.y_name_)
                    y = y.astype(self.y_dtype_)
            else:
                y = pd.Series(y, name=self.y_name_)
                y = y.astype(self.y_dtype_)
        return X, y

    def fit(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                final_estimator_time_start = time()
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)
                self.final_estimator_fit_time_ = time() - final_estimator_time_start
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            final_estimator_time_start = time()
            if hasattr(last_step, "fit_transform"):
                r = last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                r = last_step.fit(Xt, yt, **fit_params_last_step).transform(Xt)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
            return r

    def fit_resample(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            final_estimator_time_start = time()
            if hasattr(last_step, "fit_resample"):
                r = last_step.fit_resample(Xt, yt, **fit_params_last_step)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
            return r

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_predict(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            final_estimator_time_start = time()
            y_pred = self.steps[-1][-1].fit_predict(Xt, yt, **fit_params_last_step)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
        return y_pred

    @if_delegate_has_method(delegate="_final_estimator")
    def predict(self, X, **predict_params):
        X, _ = self._convert_to_df_if_needed(X)
        return super().predict(X=X, **predict_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_proba(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().predict_proba(X=X)

    @if_delegate_has_method(delegate="_final_estimator")
    def decision_function(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().decision_function(X=X)

    @if_delegate_has_method(delegate="_final_estimator")
    def score_samples(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().score_samples(X=X)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_log_proba(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().predict_log_proba(X=X)

    def _transform(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super()._transform(X)

    def _inverse_transform(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super()._inverse_transform(X)

    @if_delegate_has_method(delegate="_final_estimator")
    def score(self, X, y=None, sample_weight=None):
        X, y = self._convert_to_df_if_needed(X, y)
        return super().score(X=X, y=y, sample_weight=sample_weight)


class Pipeline(Flow):
    _component_class = BasePipeline

    _default_parameters = {
        "memory": None,
        "verbose": False,
    }

    @property
    def components_name(self) -> str:
        return "steps"

    def get_default_components_configuration(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ):
        steps = [
            (
                name,
                get_single_component_from_iterable(
                    step, pipeline_config=pipeline_config, current_stage=current_stage
                ),
            )
            for name, step in self.components
            if is_component_valid_iterable(
                step, pipeline_config=pipeline_config, current_stage=current_stage
            )
        ]
        return steps

    def __call__(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
    ):
        params = self.final_parameters.copy()
        steps = self.get_default_components_configuration(
            pipeline_config=pipeline_config,
            current_stage=current_stage,
        )
        steps = [
            (
                name,
                component(
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                    random_state=random_state,
                ),
            )
            for name, component in steps
        ]
        params["steps"] = steps

        return self._component_class(**params)

    def get_valid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        steps = self.components
        steps = [
            (
                name,
                recursively_remove_invalid_components(
                    step, pipeline_config=pipeline_config, current_stage=current_stage
                ),
            )
            for name, step in steps
            if is_component_valid_iterable(
                step, pipeline_config=pipeline_config, current_stage=current_stage
            )
        ]
        return steps

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = super().get_tuning_grid(use_extended=use_extended)
        step_grids = {
            name: get_step_choice_grid(step, use_extended=use_extended)
            for name, step in self.components
        }
        return {**step_grids, **default_grid}

    def call_tuning_grid_funcs(self, config: ComponentConfig, stage: AutoMLStage):
        super().call_tuning_grid_funcs(config, stage)
        for name, step in self.components:
            recursively_call_tuning_grid_funcs(step, config=config, stage=stage)

    def __copy__(self):
        # self.spam is to be ignored, it is calculated anew for the copy
        # create a new copy of ourselves *reusing* self.bar
        new = type(self)(tuning_grid=self.tuning_grid, **self.parameters)
        new.components = self.components.copy()
        new.components = [
            (
                copy(name),
                copy(step) if isinstance(step, (list, dict, tuple, Flow)) else step,
            )
            for name, step in new.components
        ]
        return new


class TopPipeline(Pipeline):
    def get_estimator_distribution(self):
        return (self.components[-1][0], CategoricalDistribution(self.components[-1][1]))

    def get_preprocessor_distribution(self, use_extended: bool = False) -> dict:
        grid = self.get_tuning_grid(use_extended=use_extended)
        grid.pop(self.components[-1][0])
        return {
            k: CategoricalDistribution(v) for k, v in convert_tuning_grid(grid).items()
        }

    def get_all_distributions(self, use_extended: bool = False) -> dict:
        grid = self.get_tuning_grid(use_extended=use_extended)
        return {
            k: CategoricalDistribution(v) for k, v in convert_tuning_grid(grid).items()
        }

    def remove_invalid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        super().remove_invalid_components(
            pipeline_config=pipeline_config, current_stage=current_stage
        )
        self._remove_invalid_preset_components_configurations(
            pipeline_config=pipeline_config, current_stage=current_stage
        )
        return self

    def _remove_invalid_preset_components_configurations(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ) -> list:
        converted_configurations = []
        for configuration in self.preset_configurations:
            estimator, estimator_parameters = configuration["Estimator"]
            if not estimator.is_component_valid(
                config=pipeline_config, stage=current_stage
            ):
                continue

            if pipeline_config is None:
                config = ComponentConfig(estimator=estimator)
            else:
                config = copy(pipeline_config)
                config.estimator = estimator

            hyperparams = {}
            components_to_remove = []

            for name, component_parameters in configuration.items():
                component, parameters = component_parameters
                if component != estimator and not component.is_component_valid(
                    config=pipeline_config, stage=current_stage
                ):
                    components_to_remove.append(name)
                    continue
                default_hyperparameters = {
                    k: v.default for k, v in component.get_tuning_grid().items()
                }

                diff = set(parameters.keys()) - set(default_hyperparameters.keys())
                if diff:
                    raise KeyError(
                        f"parameter names {list(diff)} are not present in _default_tuning_grid for {component}"
                    )

                parameters = {**default_hyperparameters, **parameters}
                parameters = {
                    component.get_hyperparameter_key_suffix(
                        name, parameter_name
                    ): parameter
                    for parameter_name, parameter in parameters.items()
                }
                hyperparams = {**hyperparams, **parameters}

                configuration[name] = component

            for name in components_to_remove:
                configuration.pop(name)
            configuration = {**configuration, **hyperparams}
            if configuration not in converted_configurations:
                converted_configurations.append(configuration)
        self.preset_configurations = converted_configurations

    def _convert_duplicates_in_steps_to_extra_configs(self):
        self.extra_configs = defaultdict(dict)
        for i, name_step_pair in enumerate(self.parameters[self.components_name]):
            name, step = name_step_pair
            if not isinstance(step, list):
                continue
            no_dups_step = []
            for component in step:
                if component._allow_duplicates:
                    no_dups_step.append(component)
                elif not any(isinstance(component, type(x)) for x in no_dups_step):
                    no_dups_step.append(component)
                else:
                    if type(component) not in self.extra_configs[name]:
                        self.extra_configs[name][type(component)] = []
                    self.extra_configs[name][type(component)].append(component)
            self.parameters[self.components_name][i] = (name, no_dups_step)

    def __init__(
        self,
        tuning_grid=None,
        preset_configurations: Optional[List[dict]] = None,
        **parameters,
    ) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}
        self.preset_configurations = preset_configurations or []
        assert "steps" in self.parameters
        assert "Estimator" == self.parameters["steps"][-1][0]
        self._convert_duplicates_in_steps_to_extra_configs()
