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
        "min_weight_fraction_leaf": 1e-10,
        "max_features": 0.2,
        "random_state": 0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 1e-10,
        "class_weight": "balanced",
        "ccp_alpha": 0.0,
    }

    _default_tuning_grid = {
        "criterion": CategoricalDistribution(["gini", "entropy"], cost_related=False),
        "max_depth": IntUniformDistribution(2, 15, cost_bounds="upper"),
        "min_samples_split": IntUniformDistribution(2, 20, cost_related=False),
        "min_samples_leaf": IntUniformDistribution(1, 20, cost_related=False),
        "max_features": UniformDistribution(0.1, 1.0, cost_bounds="upper"),
    }
    _default_tuning_grid_extended = {
        "min_impurity_decrease": UniformDistribution(
            1e-10, 0.5, log=True, cost_related=False
        ),
        "min_weight_fraction_leaf": UniformDistribution(
            1e-10, 0.5, log=True, cost_related=False
        ),
        "class_weight": CategoricalDistribution([None, "balanced"], cost_related=False),
    }

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _component_level = (
        ComponentLevel.RARE
    )  # uncommon because RFs are almost always better


class DecisionTreeRegressor(TreeEstimator):
    _component_class = _DecisionTreeRegressor

    _default_parameters = {
        "criterion": "mse",
        "splitter": "best",
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 1e-10,
        "max_features": 1.0,
        "random_state": 0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 1e-10,
        "ccp_alpha": 0.0,
    }

    _default_tuning_grid = {
        "criterion": CategoricalDistribution(
            ["mse", "friedman_mse", "mae", "poisson"], cost_related=False
        ),
        "max_depth": IntUniformDistribution(2, 15, cost_bounds="upper"),
        "min_samples_split": IntUniformDistribution(2, 20, cost_related=False),
        "min_samples_leaf": IntUniformDistribution(1, 20, cost_related=False),
        "max_features": UniformDistribution(0.1, 1.0, cost_bounds="upper"),
    }
    _default_tuning_grid_extended = {
        "min_impurity_decrease": UniformDistribution(
            1e-10, 0.5, log=True, cost_related=False
        ),
        "min_weight_fraction_leaf": UniformDistribution(
            1e-10, 0.5, log=True, cost_related=False
        ),
    }

    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = (
        ComponentLevel.RARE
    )  # uncommon because RFs are almost always better
