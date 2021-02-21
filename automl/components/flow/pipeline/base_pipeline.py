from copy import deepcopy
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

    def __call__(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        return_prefix_mixin: bool = False,
    ):
        params = deepcopy(self.final_parameters)
        steps = [
            (
                name,
                get_single_component_from_iterable(
                    step, pipeline_config=pipeline_config, current_stage=current_stage
                )(pipeline_config=pipeline_config, current_stage=current_stage, return_prefix_mixin=return_prefix_mixin),
            )
            for name, step in params["steps"]
            if is_component_valid_iterable(
                step, pipeline_config=pipeline_config, current_stage=current_stage
            )
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

    def __init__(self, tuning_grid=None, **parameters) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}
        assert "steps" in self.parameters
        assert len(self.parameters["steps"]) == 2
        assert "Estimator" == self.parameters["steps"][-1][0]
        assert "Preprocessor" == self.parameters["steps"][0][0]
