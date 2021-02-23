from sklearn.preprocessing import StandardScaler as _StandardScaler

from ...estimators.tree.tree_estimator import TreeEstimator
from ..transformer import Transformer, DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

class PandasStandardScaler(PandasDataFrameTransformerMixin, _StandardScaler):
    pass

class StandardScaler(Transformer):
    _component_class = PandasStandardScaler
    _default_parameters = {
        "copy": True,
        "with_mean" : True,
        "with_std": True,
    }
    _allowed_dtypes = {
        DataType.NUMERIC
    }
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config, stage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (config.estimator is None or not isinstance(config.estimator, TreeEstimator))
