from .transformer import Transformer
from ..component import ComponentLevel
from ...search.stage import AutoMLStage


class Passthrough(Transformer):
    _component_class = None
    _component_level = ComponentLevel.NECESSARY

    def __call__(
        self,
        pipeline_config: dict = None,
        current_stage: AutoMLStage = AutoMLStage.PREPROCESSING,
        random_state=None,
        return_prefix_mixin: bool = False,
    ):
        return "passthrough"
