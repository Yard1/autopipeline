from sklearn.neighbors import KNeighborsTransformer

from ..transformer import DataType
from ...component import ComponentLevel
from ..transformer import Transformer
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.knn.knn_estimator import KNNEstimator
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionDistribution,
)


class KNNTransformer(Transformer):
    _component_class = KNeighborsTransformer
    _default_parameters = {
        "mode": "distance",
        "n_neighbors": 40,
        "algorithm": "auto",
        "leaf_size": 30,
        "metric": "minkowski",
        "p": 2,
        "metric_params": None,
        "n_jobs": 1,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.COMMON

    _default_tuning_grid = {
        "p": IntUniformDistribution(1, 2),
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None or isinstance(config.estimator, KNNEstimator)
        )
