from .transformer import Transformer
from ..component import ComponentLevel, ComponentConfig
from ...search.stage import AutoMLStage
from ...search.distributions import UniformDistribution

from automl_models.components.transformers.misc.category_coalescer import (
    PandasCategoryCoalescer,
)

UPPER_BOUNDS = 0.2


class CategoryCoalescer(Transformer):
    _component_class = PandasCategoryCoalescer
    _default_parameters = {
        "minimum_fraction": 0.0001
    }
    _default_tuning_grid = {
        "minimum_fraction": UniformDistribution(0.0001, UPPER_BOUNDS, log=True, cost_related=False),
    }
    _component_level = ComponentLevel.COMMON

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        if not super_check:
            return False
        has_cat_below_upper_bound = False
        if config.X is None:
            has_cat_below_upper_bound = True
        elif not config.X.select_dtypes("category").empty:
            for col in config.X.select_dtypes("category"):
                value_counts = config.X[col].value_counts(normalize=True)
                value_counts_below_fraction = value_counts[
                    (value_counts < UPPER_BOUNDS) & (value_counts > 0)
                ]
                if not value_counts_below_fraction.empty:
                    has_cat_below_upper_bound = True
                    break
        return super_check and has_cat_below_upper_bound
