import numpy as np

from automl_models.components.estimators.neural_network import (
    FTTransformerRegressor as _FTTransformerRegressor,
    FTTransformerClassifier as _FTTransformerClassifier,
)
from .neural_network_estimator import NeuralNetworkEstimator
from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionParameter,
)
from ...component import ComponentLevel
from ....problems import ProblemType


class FTTransformerClassifier(NeuralNetworkEstimator):
    _component_class = _FTTransformerClassifier
    _has_own_cat_encoding = True

    _default_parameters = {
        "early_stopping": True,
        "random_state": 0,
        "lr": 1e-4,
        "optimizer__weight_decay": 1e-5,
        "max_epochs": 30,
        "batch_size": 256,
        "verbose": 0,
        "device": "cpu",
        "n_iter_no_change": 5,
        "iterator_train__shuffle": True,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _component_level = ComponentLevel.RARE


class FTTransformerRegressor(NeuralNetworkEstimator):
    _component_class = _FTTransformerRegressor
    _has_own_cat_encoding = True

    _default_parameters = {
        "early_stopping": True,
        "random_state": 0,
        "lr": 1e-4,
        "optimizer__weight_decay": 1e-5,
        "max_epochs": 30,
        "batch_size": 256,
        "verbose": 0,
        "device": "cpu",
        "n_iter_no_change": 5,
        "iterator_train__shuffle": True,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = ComponentLevel.RARE
