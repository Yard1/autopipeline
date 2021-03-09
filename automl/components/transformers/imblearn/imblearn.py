from ..transformer import Transformer
from ...component import ComponentConfig
from ....search.stage import AutoMLStage


class ImblearnSampler(Transformer):
    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        if not super_check or config.y is None:
            return super_check

        counts = config.y.value_counts()
        max = counts[0]
        min = counts[-1]
        # pretty conservative here, ratio could probably be higher
        return super_check and (max / min) >= 1.5
