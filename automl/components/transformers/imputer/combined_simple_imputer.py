from automl.components.transformers.transformer import Transformer
from .imputer import Imputer
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage

from ....search.distributions import CategoricalDistribution, FunctionDistribution

from automl_models.components.transformers.imputer.combined_simple_imputer import (
    PandasCombinedSimpleImputer,
)


def get_numeric_strategy(config, space):
    X = config.X
    if X is None:
        return CategoricalDistribution(["mean", "median"])
    missing_values = X.select_dtypes("number").isna().sum().sum()
    if missing_values == 0:
        return CategoricalDistribution(["median"])
    return CategoricalDistribution(["mean", "median"])


def get_categorical_strategy(config, space):
    X = config.X
    if X is None:
        return CategoricalDistribution(["most_frequent", "constant"])
    missing_values = X.select_dtypes("category").isna().sum().sum()
    if missing_values == 0:
        return CategoricalDistribution(["most_frequent"])
    return CategoricalDistribution(["most_frequent", "constant"])


class CombinedSimpleImputer(Imputer):
    _component_class = PandasCombinedSimpleImputer
    _default_parameters = {
        "numeric_strategy": "median",
        "numeric_fill_value": 0,
        "categorical_strategy": "most_frequent",
        "categorical_fill_value": "missing_value",
        "verbose": 0,
        "copy": True,
        "n_jobs": 1,
        "transformer_weights": None,
    }
    _default_tuning_grid = {
        "numeric_strategy": FunctionDistribution(get_numeric_strategy),
        "categorical_strategy": FunctionDistribution(get_categorical_strategy),
    }
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        return Transformer.is_component_valid(self, config, stage)