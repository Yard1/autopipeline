from .svm import SVM
from ....component import ComponentLevel
from .....problems import ProblemType
from .....search.distributions import UniformDistribution, CategoricalDistribution

from automl_models.components.estimators.linear_model.svm.linear_svm import (
    LinearSVRDynamicDual,
    LinearSVCCombinedPenaltyLossDynamicDual,
)


class LinearSVC(SVM):
    _component_class = LinearSVCCombinedPenaltyLossDynamicDual

    _default_parameters = {
        "penalty_loss": "l2-squared_hinge",
        "tol": 1e-4,
        "C": 1.0,
        "multi_class": "ovr",
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": "balanced",
        "verbose": 0,
        "random_state": 0,
        "max_iter": 200,
    }

    _default_tuning_grid = {
        "penalty_loss": CategoricalDistribution(
            ["l2-squared_hinge", "l1-squared_hinge", "l2-hinge"], cost_related=False
        ),
        "C": UniformDistribution(0.01, 10, cost_related=False),
    }
    _default_tuning_grid_extended = {
        "class_weight": CategoricalDistribution(["balanced", None], cost_related=False)
    }

    _component_level = ComponentLevel.UNCOMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class LinearSVR(SVM):
    _component_class = LinearSVRDynamicDual

    _default_parameters = {
        "epsilon": 0.001,
        "tol": 1e-4,
        "C": 1.0,
        "loss": "epsilon_insensitive",
        "fit_intercept": True,
        "intercept_scaling": 1,
        "verbose": 0,
        "random_state": 0,
        "max_iter": 200,
    }

    _default_tuning_grid = {
        "loss": CategoricalDistribution(
            ["epsilon_insensitive", "squared_epsilon_insensitive"], cost_related=False
        ),
        "C": UniformDistribution(
            0.01, 10, cost_related=False
        ),  # consider log like in autosklearn?
        "epsilon": UniformDistribution(0.001, 1, log=True, cost_related=False),
    }
    _default_tuning_grid_extended = {}

    _component_level = ComponentLevel.UNCOMMON
    _problem_types = {ProblemType.REGRESSION}
