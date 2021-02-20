from sklearn.linear_model import LinearRegression as _LinearRegression
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType


class LinearRegression(LinearModelEstimator):
    _component_class = _LinearRegression

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

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.REGRESSION,
    }