from automl.search.distributions.distributions import CategoricalDistribution
from sklearn.linear_model import (
    SGDClassifier as _SGDClassifier,
    SGDRegressor as _SGDRegressor,
)
from .svm import SVM
from .....problems import ProblemType
from .....search.distributions import UniformDistribution


class SGDClassifier(SVM):
    _component_class = _SGDClassifier

    _default_parameters = {
        "loss": "hinge",
        "penalty": "elasticnet",
        "alpha": 0.0001,
        "l1_ratio": 0,
        "fit_intercept": True,
        "max_iter": 10000,
        "tol": 1e-3,
        "shuffle": True,
        "verbose": 0,
        "epsilon": 0.1,
        "n_jobs": None,
        "random_state": 0,
        "learning_rate": "optimal",
        "eta0": 0.0,
        "power_t": 0.5,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "class_weight": None,
        "warm_start": False,
        "average": False,
    }

    _default_tuning_grid = {
        "C": CategoricalDistribution(
            ["hinge", "modified_huber", "squared_hinge", "perceptron"]
        ),
        "alpha": UniformDistribution(1e-7, 1e-1, log=True),
        "l1_ratio": UniformDistribution(0, 1),
        "learning_rate": CategoricalDistribution(["optimal", "invscaling", "constant"]),
        # epsilon,
        # eta0
        # power_t
    }
    _default_tuning_grid_extended = {
        "class_weight": CategoricalDistribution(
            [None, "balanced", "balanced_subsample"]
        ),
    }

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}
