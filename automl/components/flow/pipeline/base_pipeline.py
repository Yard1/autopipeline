from typing import Iterable
from copy import deepcopy
from sklearn.pipeline import Pipeline as _Pipeline, make_pipeline

from ..flow import Flow
from ..column_transformer import ColumnTransformer
from ..utils import (
    recursively_remove_invalid_components,
    get_single_component_from_iterable,
    is_component_valid_iterable,
    get_step_choice_grid,
)
from ...transformers import *
from ...estimators import *
from ...component import ComponentLevel, ComponentConfig
from ....problems import ProblemType
from ....search.stage import AutoMLStage


class BasePipeline(_Pipeline):
    pass


class Pipeline(Flow):
    _component_class = BasePipeline

    @property
    def components_name(self) -> str:
        return "steps"

    def __call__(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ):
        params = deepcopy(self.final_parameters)
        steps = [
            (
                name,
                get_single_component_from_iterable(
                    step, pipeline_config=pipeline_config, current_stage=current_stage
                )(pipeline_config=pipeline_config, current_stage=current_stage),
            )
            for name, step in params["steps"]
            if is_component_valid_iterable(
                step, pipeline_config=pipeline_config, current_stage=current_stage
            )
        ]
        params["steps"] = steps

        return self._component_class(**params)

    def remove_invalid_components(
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
        self.components = steps

        return self

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = super().get_tuning_grid(use_extended=use_extended)
        step_grids = {
            name: get_step_choice_grid(step)
            for name, step in self.final_parameters["steps"]
        }
        return {**step_grids, **default_grid}
