from typing import Iterable
from numpy import isin
from .flow import Flow
from ..component import ComponentConfig
from ...search.stage import AutoMLStage

def get_step_choice_grid(step):
    if isinstance(step, Iterable):
        grid = [(substep, substep.get_tuning_grid()) for substep in step]
    else:
        grid = step.get_tuning_grid()
    return grid


def append_components_name_if_possible(name:str, flow:Flow) -> str:
    try:
        return f"{name}__{flow.components_name}"
    except:
        return name


def recursively_remove_invalid_components(
    component, pipeline_config: ComponentConfig, current_stage: AutoMLStage
):
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
    if isinstance(component, Iterable):
        return next(
            y
            for y in sorted(component, key=lambda x: x._component_level)
            if y.is_component_valid(config=pipeline_config, stage=current_stage)
        )
    return component


def is_component_valid_iterable(
    component, pipeline_config: ComponentConfig, current_stage: AutoMLStage
):
    if isinstance(component, Iterable):
        return any(
            x.is_component_valid(config=pipeline_config, stage=current_stage)
            for x in component
        )
    return component.is_component_valid(config=pipeline_config, stage=current_stage)