from .transformer import Transformer
from ..component import ComponentLevel

from automl_models.components.transformers.misc.unknown_categories_dropper import (
    PandasUnknownCategoriesDropper,
)


class UnknownCategoriesDropper(Transformer):
    _component_class = PandasUnknownCategoriesDropper
    _default_parameters = {}
    _default_tuning_grid = {}
    _component_level = ComponentLevel.NECESSARY
