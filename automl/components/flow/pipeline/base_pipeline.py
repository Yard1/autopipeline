from typing import List, Optional

from copy import copy
from sklearn.pipeline import Pipeline as _Pipeline

from ..flow import Flow
from ..utils import (
    recursively_remove_invalid_components,
    get_single_component_from_iterable,
    is_component_valid_iterable,
    get_step_choice_grid,
)
from ..utils import convert_tuning_grid
from ...transformers import *
from ...estimators import *
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ....search.distributions import CategoricalDistribution


class BasePipeline(_Pipeline):
    pass


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
        return_prefix_mixin: bool = False,
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
                    return_prefix_mixin=return_prefix_mixin,
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
            name: get_step_choice_grid(step)
            for name, step in self.final_parameters["steps"]
        }
        return {**step_grids, **default_grid}


class TopPipeline(Pipeline):
    def get_estimator_distribution(self):
        return (self.components[-1][0], CategoricalDistribution(self.components[-1][1]))

    def get_preprocessor_distribution(self):
        grid = self.get_tuning_grid()
        grid.pop(self.components[-1][0])
        return {
            k: CategoricalDistribution(v) for k, v in convert_tuning_grid(grid).items()
        }

    def get_all_distributions(self):
        grid = self.get_tuning_grid()
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
                    f"{name}__{component.prefix}_{parameter_name}": parameter
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
        assert len(self.parameters["steps"]) == 2
        assert "Estimator" == self.parameters["steps"][-1][0]
        assert "Preprocessor" == self.parameters["steps"][0][0]
