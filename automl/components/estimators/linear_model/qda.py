from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _QuadraticDiscriminantAnalysis
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType
from ...component import ComponentLevel


class QuadraticDiscriminantAnalysis(LinearModelEstimator):
    _component_class = _QuadraticDiscriminantAnalysis

    _default_parameters = {
        "priors": None,
        "reg_param": 0.0,
        "store_covariance": False,
        "tol": 0.0001
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}

    _component_level = ComponentLevel.UNCOMMON
