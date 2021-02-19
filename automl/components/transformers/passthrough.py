from .transformer import Transformer
from ..component import ComponentLevel
from ...search.stage import AutoMLStage

class Passthrough(Transformer):
    _component_class = None
    _component_level = ComponentLevel.NECESSARY
    def __call__(self, pipeline_config: dict, current_stage: AutoMLStage):
        return "passthrough"