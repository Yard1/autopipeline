from abc import ABC
from typing import Optional
from enum import IntEnum
from ..problems import ProblemType
from ..search.stage import AutoMLStage
from .compatibility.mixin import PrefixParamsMixin


class ComponentLevel(IntEnum):
    NECESSARY = 1
    COMMON = 2
    UNCOMMON = 3
    RARE = 4


class ComponentConfig:
    def __init__(
        self,
        level: Optional[ComponentLevel] = None,
        problem_type: Optional[ProblemType] = None,
        missing_values: Optional[bool] = None,
        estimator: Optional[type] = None,
        component: Optional[type] = None,
    ) -> None:
        self.level = level
        self.problem_type = problem_type
        self.missing_values = missing_values
        self.component = component
        self.estimator = estimator


class Component(ABC):
    _component_class = None
    _component_class_prefix = None

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

        for k in self._default_tuning_grid.keys():
            if k not in self._default_parameters:
                raise KeyError(
                    f"_default_parameters is missing key {k} present in _default_tuning_grid"
                )
            self._default_tuning_grid[k].default = self._default_parameters[k]

        try:
            self._component_class_prefix = type(
                self._component_class.__name__,
                (PrefixParamsMixin, self._component_class),
                {"_automl_prefix": self.prefix},
            )
        except AttributeError:
            self._component_class_prefix = self._component_class

    def __call__(
        self,
        pipeline_config: dict = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
        return_prefix_mixin: bool = False,
    ):
        params = self.final_parameters

        if "random_state" in self._default_parameters:
            params["random_state"] = random_state

        if return_prefix_mixin:
            return self._component_class_prefix(**params)
        return self._component_class(**params)

    @property
    def final_parameters(self):
        return {
            **self._default_parameters,
            **self.parameters,
        }

    @property
    def prefix(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.final_parameters.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_tuning_grid = (
            self._default_tuning_grid_extended
            if use_extended
            else self._default_tuning_grid
        )
        return {**default_tuning_grid, **self.tuning_grid}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        is_level_good = config.level is None or (self._component_level <= config.level)
        is_problem_type_good = config.problem_type is None or (
            config.problem_type in self._problem_types
        )

        return is_level_good and is_problem_type_good
