from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier, DecisionTreeRegressor as _DecisionTreeRegressor
from .tree_estimator import TreeEstimator
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
        "max_features": "sqrt", # TODO: make dynamic
        "random_state": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }