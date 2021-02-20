from sklearn.linear_model import LogisticRegression as _LogisticRegression
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType
from ....search.distributions import *

class LogisticRegression(LinearModelEstimator):
    _component_class = _LogisticRegression

    _default_parameters = {
        "penalty": "elasticnet",
        "dual": False,
        "tol": 1e-4,
        "C": 1.0,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "random_state": None,
        "solver": "saga",
        "max_iter": 1000,
        "multi_class": "auto",
        "verbose": 0,
        "warm_start": False,
        "n_jobs": None,
        "l1_ratio": 0,
    }

    _default_tuning_grid = {
        "C": UniformDistribution(0.01, 10),
        "l1_ratio": UniformDistribution(0, 1),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS
    }