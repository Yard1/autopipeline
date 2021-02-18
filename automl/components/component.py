from abc import ABC, abstractmethod


class Component(ABC):
    _component_class = None
    
    _default_parameters = {}
    
    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    def __init__(self, **parameters) -> None:
        self.parameters = parameters

    def __call__(self):
        params = {
            **self._default_parameters,
            **self.parameters,
        }
        return self._component_class(**params)

    @classmethod
    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        return (
            self._default_tuning_grid_extended
            if use_extended
            else self._default_tuning_grid
        )
