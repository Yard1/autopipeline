from sklearn.linear_model import (
    LinearRegression as _LinearRegression,
    ElasticNet as _ElasticNet,
    ElasticNetCV as _ElasticNetCV,
)
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType
from ....search.distributions import UniformDistribution, CategoricalDistribution


class LinearRegression(LinearModelEstimator):
    _component_class = _LinearRegression

    _default_parameters = {
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
        "n_jobs": None,
        "positive": False,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.REGRESSION,
    }


class ElasticNet(LinearModelEstimator):
    _component_class = _ElasticNet

    _default_parameters = {
        "alpha": 1.0,
        "l1_ratio": 1,
        "fit_intercept": True,
        "normalize": False,
        "precompute": False,
        "max_iter": 1000,
        "tol": 1e-4,
        "warm_start": False,
        "positive": False,
        "random_state": None,
        "selection": "random",
        "copy_X": True,
    }

    _default_tuning_grid = {
        "alpha": UniformDistribution(0.01, 10, cost_related=False),
        "l1_ratio": UniformDistribution(0, 1, cost_related=False),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.REGRESSION,
    }


class ElasticNetCV(LinearModelEstimator):
    _component_class = _ElasticNetCV

    _default_parameters = {
        "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        "eps": 1e-3,
        "n_alphas": 100,
        "alphas": None,
        "fit_intercept": True,
        "normalize": False,
        "precompute": False,
        "max_iter": 1000,
        "tol": 1e-4,
        "cv": 5,
        "verbose": 0,
        "n_jobs": None,
        "positive": False,
        "random_state": None,
        "selection": "random",
        "copy_X": True,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.REGRESSION,
    }