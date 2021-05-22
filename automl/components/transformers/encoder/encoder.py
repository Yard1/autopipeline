from ..transformer import Transformer
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage

class Encoder(Transformer):
    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (config.X is None or list(config.X.select_dtypes("category")))

