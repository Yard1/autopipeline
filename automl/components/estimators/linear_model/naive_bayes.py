from sklearn.naive_bayes import GaussianNB as _GaussianNB
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType
from ...component import ComponentLevel


class GaussianNB(LinearModelEstimator):
    _component_class = _GaussianNB

    _default_parameters = {
        "priors": None,
        "var_smoothing": 1e-9,
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}

    _component_level = ComponentLevel.UNCOMMON
