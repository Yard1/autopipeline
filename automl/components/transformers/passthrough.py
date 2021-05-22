from typing import Callable, Optional
from .transformer import Transformer
from ..component import ComponentLevel, ComponentConfig
from ...search.stage import AutoMLStage


class Passthrough(Transformer):
    _component_class = None
    _component_level = ComponentLevel.NECESSARY
    _allow_duplicates = True

    def __init__(
        self,
        validity_condition: Optional[Callable] = None,
        tuning_grid=None,
        **parameters
    ) -> None:
        self.validity_condition = validity_condition
        super().__init__(tuning_grid=tuning_grid, **parameters)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.validity_condition})"

    def __call__(
        self,
        pipeline_config: dict = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
    ):
        return "passthrough"

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if self.validity_condition is not None:
            return self.validity_condition(config=config, stage=stage)
        return super().is_component_valid(config, stage)
