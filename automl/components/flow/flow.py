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

    def get_default_components_configuration(
        self,
        pipeline_config: ComponentConfig = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
    ):
        raise NotImplementedError()

    def remove_invalid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        self.components = self.get_valid_components(
            pipeline_config=pipeline_config, current_stage=current_stage
        )

        return self

    def get_valid_components(
        self, pipeline_config: ComponentConfig, current_stage: AutoMLStage
    ):
        raise NotImplementedError()

    def __copy__(self):
        # self.spam is to be ignored, it is calculated anew for the copy
        # create a new copy of ourselves *reusing* self.bar
        new = type(self)(tuning_grid=self.tuning_grid, **self.parameters)
        new.components = self.components.copy()
        return new
