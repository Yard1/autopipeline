import numpy as np
from automl.search.distributions.distributions import FunctionParameter
from .tree_estimator import TreeEstimator
from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    FunctionDistribution,
)
from .utils import estimate_max_depth
from ...component import ComponentLevel
from ....problems import ProblemType

from automl_models.components.estimators.tree.random_forest import (
    RandomForestExtraTreesClassifier,
    RandomForestExtraTreesRegressor,
)


def get_rf_n_estimators(config, space):
    X = config.X
    if X is None:
        return IntUniformDistribution(10, 100, log=True)
    return IntUniformDistribution(10, min(2048, int(X.shape[0])), log=True)


def get_rf_sqrt_features(config, space):
    X = config.X
    return np.sqrt(X.shape[1]) / X.shape[1]


class RandomForestClassifier(TreeEstimator):
    _component_class = RandomForestExtraTreesClassifier

    _default_parameters = {
        "n_estimators": 100,
        "randomization_type": "rf",
        "criterion": "gini",
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 1e-10,
        "max_features": FunctionParameter(get_rf_sqrt_features),
        "random_state": 0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 1e-10,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": None,
        "class_weight": "balanced",
        "ccp_alpha": 0.0,
        "verbose": 0,
        "warm_start": False,
        "max_samples": None,
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(get_rf_n_estimators, cost_bounds="upper"),
        "randomization_type": CategoricalDistribution(["rf", "et"], cost_related=False),
        "criterion": CategoricalDistribution(["gini", "entropy"], cost_related=False),
        "min_samples_split": IntUniformDistribution(2, 20, cost_related=False),
        "min_samples_leaf": IntUniformDistribution(1, 20, cost_related=False),
        "max_features": UniformDistribution(0.1, 1.0, cost_bounds="upper"),
    }
    _default_tuning_grid_extended = {
        "max_depth": IntUniformDistribution(2, 15, cost_bounds="upper"),
        "min_impurity_decrease": UniformDistribution(
            1e-10, 0.1, log=True, cost_related=False
        ),
        "min_weight_fraction_leaf": UniformDistribution(
            1e-10, 0.1, log=True, cost_related=False
        ),
        "class_weight": CategoricalDistribution(
            [None, "balanced", "balanced_subsample"], cost_related=False
        ),
    }

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _component_level = ComponentLevel.COMMON


class RandomForestRegressor(TreeEstimator):
    _component_class = RandomForestExtraTreesRegressor

    _default_parameters = {
        "n_estimators": 100,
        "randomization_type": "rf",
        "criterion": "mse",
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 1e-10,
        "max_features": 1.0,
        "random_state": 0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 1e-10,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": None,
        "ccp_alpha": 0.0,
        "verbose": 0,
        "warm_start": False,
        "max_samples": None,
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(get_rf_n_estimators, cost_bounds="upper"),
        "randomization_type": CategoricalDistribution(["rf", "et"], cost_related=False),
        # "criterion": CategoricalDistribution(["mse", "mae"]),
        "min_samples_split": IntUniformDistribution(2, 20, cost_related=False),
        "min_samples_leaf": IntUniformDistribution(1, 20, cost_related=False),
        "max_features": UniformDistribution(0.1, 1.0, cost_bounds="upper"),
    }
    _default_tuning_grid_extended = {
        "max_depth": IntUniformDistribution(2, 15, cost_bounds="upper"),
        "min_impurity_decrease": UniformDistribution(
            1e-10, 0.1, log=True, cost_related=False
        ),
        "min_weight_fraction_leaf": UniformDistribution(
            1e-10, 0.1, log=True, cost_related=False
        ),
    }

    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = ComponentLevel.COMMON
