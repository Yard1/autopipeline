from abc import ABC
from typing import Optional, Union
from enum import IntEnum
from collections import UserDict
from ..problems import ProblemType
from ..search.stage import AutoMLStage


class ComponentLevel(IntEnum):
    NECESSARY = 1
    COMMON = 2
    UNCOMMON = 3
    RARE = 4

    @staticmethod
    def translate(label: Union[str, int]) -> "ComponentLevel":
        if isinstance(label, ComponentLevel):
            return label
        if label in (1, "NECESSARY", "NECESSARY".lower()):
            return ComponentLevel.NECESSARY
        if label in (2, "COMMON", "COMMON".lower()):
            return ComponentLevel.COMMON
        if label in (3, "UNCOMMON", "UNCOMMON".lower()):
            return ComponentLevel.UNCOMMON
        if label in (4, "RARE", "RARE".lower()):
            return ComponentLevel.RARE
        raise ValueError(f"Cannot translate '{label}' to a ComponentLevel object!")


class ComponentConfig(UserDict):
    def __init__(self, **kwargs) -> None:
        self.data = kwargs

    def __getattr__(self, name: str):
        return self.data.get(name, None)

    def __repr__(self) -> str:
        return str(self.data)


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

    _automl_id_sign = "\u200B"

    def __init__(self, tuning_grid=None, **parameters) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}

        for k in self._default_tuning_grid.keys():
            if k not in self._default_parameters:
                raise KeyError(
                    f"_default_parameters is missing key {k} present in _default_tuning_grid"
                )
            self._default_tuning_grid[k].default = self._default_parameters[k]

        for k in self._default_tuning_grid_extended.keys():
            if k not in self._default_parameters:
                raise KeyError(
                    f"_default_parameters is missing key {k} present in _default_tuning_grid"
                )
            self._default_tuning_grid_extended[k].default = self._default_parameters[k]

    def __call__(
        self,
        pipeline_config: dict = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
    ):
        params = self.final_parameters

        if "random_state" in self._default_parameters:
            params["random_state"] = random_state

        return self._component_class(**params)

    @property
    def final_parameters(self):
        return {
            **self._default_parameters,
            **self.parameters,
        }

    @property
    def automl_id(self):
        return f"{self._automl_id_sign}{self.__class__.__name__}{self._automl_id_sign}"

    def get_hyperparameter_key_suffix(self, prefix, hyperparam_name):
        return f"{prefix}__{self.automl_id}__{hyperparam_name}"

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.final_parameters.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        return {
            **self._default_tuning_grid,
            **(self._default_tuning_grid_extended if use_extended else {}),
            **self.tuning_grid,
        }

    def call_tuning_grid_funcs(
        self, config: ComponentConfig, stage: AutoMLStage, use_extended: bool = False
    ):
        called_tuning_grid = {
            k: v(config, stage)
            for k, v in self._default_tuning_grid.items()
            if callable(v)
        }
        if use_extended:
            called_extended_tuning_grid = {
                k: v(config, stage)
                for k, v in self._default_tuning_grid_extended.items()
                if callable(v)
            }
            called_tuning_grid = {**called_tuning_grid, **called_extended_tuning_grid}
        self.tuning_grid = {**called_tuning_grid, **self.tuning_grid}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        is_level_good = config.level is None or (self._component_level <= config.level)
        is_problem_type_good = config.problem_type is None or (
            config.problem_type in self._problem_types
        )

        return is_level_good and is_problem_type_good
