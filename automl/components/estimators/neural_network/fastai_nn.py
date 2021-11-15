import numpy as np

from automl_models.components.estimators.neural_network import (
    FastAINNClassifier as _FastAINNClassifier,
    FastAINNRegressor as _FastAINNRegressor,
)
from .neural_network_estimator import NeuralNetworkEstimator
from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionParameter,
)
from .utils import get_category_cardinalities
from ...component import ComponentLevel
from ....problems import ProblemType


class FastAINNClassifier(NeuralNetworkEstimator):
    _component_class = _FastAINNClassifier
    _has_own_cat_encoding = True

    _default_parameters = {
        "early_stopping": True,
        "random_state": 0,
        "lr": 1e-3,
        "max_epochs": 100,
        "batch_size_power": 8,
        "verbose": 0,
        "device": "cpu",
        "category_cardinalities": FunctionParameter(get_category_cardinalities),
        "n_iter_no_change": 5,
        "module__layers": (200, 100),
        "module__embed_p": 0.1,
        "module__ps": 0.1,
        "iterator_train__shuffle": True,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {
        "module__layers": CategoricalDistribution(
            [
                None,
                (200, 100),
                (200,),
                (500,),
                (1000,),
                (500, 200),
                (50, 25),
                (1000, 500),
                (200, 100, 50),
                (500, 200, 100),
                (1000, 500, 200),
            ],
            cost_related=True,
        ),
        "module__embed_p": UniformDistribution(0.0, 0.5, cost_related=False),
        "module__ps": UniformDistribution(0.0, 0.5, cost_related=False),
        "batch_size_power": IntUniformDistribution(6, 12, cost_related=True),
        "lr": UniformDistribution(5e-5, 1e-1, log=True, cost_bounds="lower"),
    }

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _component_level = ComponentLevel.COMMON


class FastAINNRegressor(NeuralNetworkEstimator):
    _component_class = _FastAINNRegressor
    _has_own_cat_encoding = True

    _default_parameters = {
        "early_stopping": True,
        "random_state": 0,
        "lr": 1e-3,
        "max_epochs": 100,
        "batch_size_power": 8,
        "verbose": 0,
        "device": "cpu",
        "category_cardinalities": FunctionParameter(get_category_cardinalities),
        "n_iter_no_change": 5,
        "module__layers": (200, 100),
        "module__embed_p": 0.1,
        "module__ps": 0.1,
        "iterator_train__shuffle": True,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {
        "module__layers": CategoricalDistribution(
            [
                None,
                (200, 100),
                (200,),
                (500,),
                (1000,),
                (500, 200),
                (50, 25),
                (1000, 500),
                (200, 100, 50),
                (500, 200, 100),
                (1000, 500, 200),
            ],
            cost_related=True,
        ),
        "module__embed_p": UniformDistribution(0.0, 0.5, cost_related=False),
        "module__ps": UniformDistribution(0.0, 0.5, cost_related=False),
        "batch_size_power": IntUniformDistribution(6, 12, cost_related=True),
        "lr": UniformDistribution(5e-5, 1e-1, log=True, cost_bounds="lower"),
    }

    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = ComponentLevel.COMMON
