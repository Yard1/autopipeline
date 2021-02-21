from typing import Iterable, Union
from .flow import Flow
from ..component import ComponentConfig
from ...search.stage import AutoMLStage
from ...search.distributions import CategoricalDistribution


def get_step_choice_grid(step, return_distribution: bool = False):
    if isinstance(step, Iterable):
        grid = step if not return_distribution else CategoricalDistribution(step)
    else:
        grid = step.get_tuning_grid()
    return grid


def append_components_name_if_possible(name: str, flow: Flow) -> str:
    try:
        return f"{name}__{flow.components_name}"
    except:
        return name


def recursively_remove_invalid_components(
    component, pipeline_config: ComponentConfig, current_stage: AutoMLStage
):
    if isinstance(component, CategoricalDistribution):
        component = component.values
    if isinstance(component, Iterable):
        for subcomponent in component:
            recursively_remove_invalid_components(
                subcomponent,
                pipeline_config=pipeline_config,
                current_stage=current_stage,
            )
    elif isinstance(component, Flow):
        component.remove_invalid_components(
            pipeline_config=pipeline_config, current_stage=current_stage
        )
    return component


def get_single_component_from_iterable(
    component, pipeline_config: ComponentConfig, current_stage: AutoMLStage
):
    if isinstance(component, CategoricalDistribution):
        component = component.values
    if isinstance(component, Iterable):
        component = next(
            y
            for y in sorted(component, key=lambda x: x._component_level)
            if y.is_component_valid(config=pipeline_config, stage=current_stage)
        )
    return component


def is_component_valid_iterable(
    component, pipeline_config: ComponentConfig, current_stage: AutoMLStage
):
    if isinstance(component, CategoricalDistribution):
        component = component.values
    if isinstance(component, Iterable):
        return any(
            x.is_component_valid(config=pipeline_config, stage=current_stage)
            for x in component
        )
    return component.is_component_valid(config=pipeline_config, stage=current_stage)


def convert_tuning_grid(grid: Union[list, dict]) -> dict:
    def convert_tuning_grid_step(
        grid: Union[list, dict], param_dict: dict, level: int = 0, level_name: str = ""
    ) -> None:
        if isinstance(grid, list):
            param_dict[level_name] = grid
            return
        for k, v in grid.items():
            convert_tuning_grid_step(
                v, param_dict, level + 1, f"{level_name+'__' if level_name else ''}{k}"
            )
        return

    param_dict = {}
    convert_tuning_grid_step(grid, param_dict)
    return param_dict
