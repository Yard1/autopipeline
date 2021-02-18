from abc import ABC, abstractmethod

from ..problems import ProblemType


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

    def __init__(self, tuning_grid=None, **parameters) -> None:
        self.parameters = parameters
        self.tuning_grid = tuning_grid or {}

    def __call__(self):
        return self._component_class(**self.final_parameters)

    @property
    def final_parameters(self):
        return {
            **self._default_parameters,
            **self.parameters,
        }

    def __repr__(self) -> str:
        return self.__class__.__name__

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = (
            self._default_tuning_grid_extended
            if use_extended
            else self._default_tuning_grid
        )
        return {**default_grid, **self.tuning_grid}
