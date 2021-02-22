from sklearn.impute import SimpleImputer as _SimpleImputer

from ..transformer import Transformer, DataType
from ...component import ComponentLevel, ComponentConfig
from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ....search.distributions import CategoricalDistribution
from ....search.stage import AutoMLStage


class PandasSimpleImputer(PandasDataFrameTransformerMixin, _SimpleImputer):
    pass


class SimpleNumericImputer(Transformer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "mean",
        "fill_value": 0,
        "verbose": 0,
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": CategoricalDistribution(["mean", "median", "constant"])
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (config.X is None or config.X.isnull().values.any())


class SimpleCategoricalImputer(Transformer):
    _component_class = PandasSimpleImputer
    _default_parameters = {
        "strategy": "most_frequent",
        "fill_value": "missing_value",
        "verbose": 0,
        "copy": True,
        "add_indicator": False,
    }
    _default_tuning_grid = {
        "strategy": CategoricalDistribution(["most_frequent", "constant"])
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (config.X is None or config.X.isnull().values.any())
