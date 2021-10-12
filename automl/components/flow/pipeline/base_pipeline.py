from typing import List, Optional
from collections import defaultdict

from copy import copy

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

from automl_models.components.flow.pipeline.base_pipeline import BasePipeline


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
            if not name.startswith("target_pipeline")
        ]
        target_steps = [
            (
                name,
                component(
                    pipeline_config=pipeline_config,
                    current_stage=current_stage,
                    random_state=random_state,
                ),
            )
            for name, component in steps
            if name.startswith("target_pipeline")
        ]
        params["steps"] = steps
        if target_steps:
            params["target_pipeline"] = BasePipeline(target_steps)

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
