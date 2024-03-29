from abc import ABC
from typing import Optional, Union
from enum import IntEnum
from collections import UserDict
from ..problems import ProblemType
from ..search.stage import AutoMLStage
from ..search.distributions import FunctionParameter


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
    _consider_for_initial_combinations = True

    _allow_duplicates = False

    def __init__(self, tuning_grid=None, **parameters) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}
        self.called_default_parameters = self._default_parameters

        for k in self._default_tuning_grid.keys():
            if isinstance(self._default_parameters[k], FunctionParameter):
                continue
            if k not in self._default_parameters:
                raise KeyError(
                    f"_default_parameters is missing key {k} present in _default_tuning_grid"
                )
            self._default_tuning_grid[k].default = self._default_parameters[k]

        for k in self._default_tuning_grid_extended.keys():
            if isinstance(self._default_parameters[k], FunctionParameter):
                continue
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
    def final_parameters(self) -> dict:
        return {
            **self.called_default_parameters,
            **self.parameters,
        }

    def get_hyperparameter_key_suffix(self, prefix: str, hyperparam_name: str) -> str:
        return f"{prefix}__{self.short_name}__{hyperparam_name}"

    def __repr__(self) -> str:
        params = [f"{key}={value}" for key, value in self.final_parameters.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    @property
    def short_name(self) -> str:
        return self.__class__.__name__

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        if hasattr(self, "called_tuning_grid"):
            return {
                **self.called_tuning_grid,
                **(self.called_extended_tuning_grid if use_extended else {}),
                **self.tuning_grid,
            }
        return {
            **self._default_tuning_grid,
            **(self._default_tuning_grid_extended if use_extended else {}),
            **self.tuning_grid,
        }

    def call_tuning_grid_funcs(self, config: ComponentConfig, stage: AutoMLStage):
        self.called_default_parameters = {
            k: v(config, stage) if isinstance(v, FunctionParameter) else v
            for k, v in self._default_parameters.items()
        }
        self.called_tuning_grid = {
            k: v(config, stage) if callable(v) else v
            for k, v in self._default_tuning_grid.items()
        }
        self.called_extended_tuning_grid = {
            k: v(config, stage) if callable(v) else v
            for k, v in self._default_tuning_grid_extended.items()
        }
        for k in self.called_tuning_grid.keys():
            if k not in self.called_default_parameters:
                raise KeyError(
                    f"called_default_parameters is missing key {k} present in called_tuning_grid"
                )
            self.called_tuning_grid[k].default = self.called_default_parameters[k]

        for k in self.called_extended_tuning_grid.keys():
            if k not in self.called_default_parameters:
                raise KeyError(
                    f"called_default_parameters is missing key {k} present in called_extended_tuning_grid"
                )
            self.called_extended_tuning_grid[
                k
            ].default = self.called_default_parameters[k]

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        is_level_good = config.level is None or (self._component_level <= config.level)
        is_problem_type_good = config.problem_type is None or (
            config.problem_type in self._problem_types
        )

        return is_level_good and is_problem_type_good
