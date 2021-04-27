from sklearn.neighbors import KNeighborsTransformer

from .knn import KNN
from ..transformer import DataType
from ...component import ComponentLevel
from ....problems import ProblemType
from ....search.distributions import IntUniformDistribution

from automl_models.components.transformers.knn.knn_transformer import (
    NeighborhoodComponentsAnalysisCaching,
)


class KNNTransformer(KNN):
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
    _problem_types = {
        ProblemType.REGRESSION,
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _default_tuning_grid = {
        "p": IntUniformDistribution(1, 2, cost_related=False),
    }


class NCATransformer(KNN):
    _component_class = NeighborhoodComponentsAnalysisCaching
    _default_parameters = {
        "nca_transformer": None,
        "knn_transformer": KNNTransformer()(),
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.COMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}
