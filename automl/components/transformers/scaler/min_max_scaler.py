from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

from .scaler import Scaler
from ..transformer import DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.knn.knn_estimator import KNNEstimator


class PandasMinMaxScaler(PandasDataFrameTransformerMixin, _MinMaxScaler):
    pass


class MinMaxScaler(Scaler):
    _component_class = PandasMinMaxScaler
    _default_parameters = {
        "feature_range": (0, 1),
        "copy": True,
        "clip": False,
    }
    _allowed_dtypes = {DataType.NUMERIC}
    _component_level = ComponentLevel.NECESSARY

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, KNNEstimator)
        )