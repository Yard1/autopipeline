from ..transformers.transformer import Transformer
from ..component import ComponentLevel, ComponentConfig
from ...search.stage import AutoMLStage


class Flow(Transformer):
    _component_level = ComponentLevel.NECESSARY

    @property
    def components_name(self) -> str:
        return ""

    @property
    def components(self):
        return self.parameters[self.components_name]

    @components.setter
    def components(self, x):
        self.parameters[self.components_name] = x

    def remove_invalid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        return self
