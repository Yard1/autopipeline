import numpy as np

from sklearn.tree import (
    DecisionTreeClassifier as _DecisionTreeClassifier,
    DecisionTreeRegressor as _DecisionTreeRegressor,
)
from .tree_estimator import TreeEstimator
from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionDistribution,
)
from .utils import estimate_max_depth
from ...component import ComponentLevel
from ....problems import ProblemType


class DecisionTreeClassifier(TreeEstimator):
    _component_class = _DecisionTreeClassifier

    _default_parameters = {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",  # TODO: make dynamic
        "random_state": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
    }

    _default_tuning_grid = {
        "criterion": CategoricalDistribution(["gini", "entropy"]),
        "max_depth": FunctionDistribution(estimate_max_depth),
        "min_samples_split": IntUniformDistribution(2, 20),
        "min_samples_leaf": IntUniformDistribution(1, 20),
        "max_features": CategoricalDistribution([1.0, "sqrt", "log2"]),
    }
    _default_tuning_grid_extended = {
        "min_impurity_decrease": UniformDistribution(0.0, 0.5, log=True),
        "min_weight_fraction_leaf": UniformDistribution(0.0, 0.5, log=True),
    }

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _component_level = (
        ComponentLevel.UNCOMMON
    )  # uncommon because RFs are almost always better


class DecisionTreeRegressor(TreeEstimator):
    _component_class = _DecisionTreeRegressor

    _default_parameters = {
        "criterion": "mse",
        "splitter": "best",
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,  # TODO: make dynamic
        "random_state": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
    }

    _default_tuning_grid = {
        "criterion": CategoricalDistribution(["mse", "friedman_mse", "mae", "poisson"]),
        "max_depth": FunctionDistribution(estimate_max_depth),
        "min_samples_split": IntUniformDistribution(2, 20),
        "min_samples_leaf": IntUniformDistribution(1, 20),
        "max_features": CategoricalDistribution([1.0, "sqrt", "log2"]),
    }
    _default_tuning_grid_extended = {
        "min_impurity_decrease": UniformDistribution(0.0, 0.5, log=True),
        "min_weight_fraction_leaf": UniformDistribution(0.0, 0.5, log=True),
    }

    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = (
        ComponentLevel.UNCOMMON
    )  # uncommon because RFs are almost always better
