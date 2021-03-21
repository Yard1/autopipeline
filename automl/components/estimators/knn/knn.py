from sklearn.neighbors import (
    KNeighborsClassifier as _KNeighborsClassifier,
    KNeighborsRegressor as _KNeighborsRegressor,
)
from .knn_estimator import KNNEstimator
from ....problems import ProblemType
from ....search.distributions import IntUniformDistribution, CategoricalDistribution


# TODO: NCA
class KNeighborsClassifier(KNNEstimator):
    _component_class = _KNeighborsClassifier

    _default_parameters = {
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": None,
    }

    _default_tuning_grid = {
        "n_neighbors": IntUniformDistribution(1, 100, log=True),
        "weights": CategoricalDistribution(["uniform", "distance"]),
        "p": IntUniformDistribution(1, 2),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class KNeighborsRegressor(KNNEstimator):
    _component_class = _KNeighborsRegressor

    _default_parameters = {
        "n_neighbors": 5,
        "weights": "uniform",
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": None,
    }

    _default_tuning_grid = {
        "n_neighbors": IntUniformDistribution(1, 100, log=True),
        "weights": CategoricalDistribution(["uniform", "distance"]),
        "p": IntUniformDistribution(1, 2),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.REGRESSION}
