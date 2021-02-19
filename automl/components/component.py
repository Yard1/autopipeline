from abc import ABC, abstractmethod
from typing import Optional
from enum import IntEnum
from ..problems import ProblemType
from ..search.stage import AutoMLStage


class ComponentLevel(IntEnum):
    NECESSARY = 1
    COMMON = 2
    UNCOMMON = 3
    RARE = 4


class ComponentConfig:
    def __init__(
        self,
        level: ComponentLevel,
        problem_type: ProblemType,
        estimator: Optional[type] = None,
        component: Optional[type] = None,
    ) -> None:
        self.level = level
        self.problem_type = problem_type
        self.component = component
        self.estimator = estimator


class Component(ABC):
    _component_class = None

    _default_parameters = {}

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _forbidden_estimators = {}
    _problem_types = {
        ProblemType.REGRESSION,
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }
    _component_level = ComponentLevel.COMMON

    def __init__(self, tuning_grid=None, **parameters) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}

    def __call__(self, pipeline_config: dict = None, current_stage: AutoMLStage = AutoMLStage.PREPROCESSING):
        return self._component_class(**self.final_parameters)

    @property
    def final_parameters(self):
        return {
            **self._default_parameters,
            **self.parameters,
        }

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.final_parameters.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = (
            self._default_tuning_grid_extended
            if use_extended
            else self._default_tuning_grid
        )
        return {**default_grid, **self.tuning_grid}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        is_level_good = self._component_level <= config.level
        is_problem_type_good = config.problem_type in self._problem_types

        return is_level_good and is_problem_type_good
