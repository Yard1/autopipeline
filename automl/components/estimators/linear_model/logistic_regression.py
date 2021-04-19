import numpy as np
from sklearn.linear_model import (
    LogisticRegression as _LogisticRegression,
    LogisticRegressionCV as _LogisticRegressionCV,
)
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType
from ....search.distributions import UniformDistribution, CategoricalDistribution


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
        "random_state": 0,
        "solver": "saga",
        "max_iter": 200,
        "multi_class": "auto",
        "verbose": 0,
        "warm_start": False,
        "n_jobs": None,
        "l1_ratio": 0,
    }

    _default_tuning_grid = {
        "C": UniformDistribution(0.01, 10, cost_related=False),
        "l1_ratio": UniformDistribution(0, 1, cost_related=False),
        "class_weight": CategoricalDistribution([None, "balanced"], cost_related=False),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class LogisticRegressionCV(LinearModelEstimator):
    _component_class = _LogisticRegressionCV

    _default_parameters = {
        "penalty": "elasticnet",
        "dual": False,
        "tol": 1e-4,
        "Cs": 10,
        "cv": 5,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "random_state": 0,
        "scoring": None,
        "solver": "saga",
        "max_iter": 200,
        "multi_class": "auto",
        "verbose": 0,
        "n_jobs": None,
        "l1_ratios": [0, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        "refit": True,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}
