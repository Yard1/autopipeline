from sklearn.linear_model import LinearRegression as _LinearRegression
from .linear_model_estimator import LinearModelEstimator
from ....problems import ProblemType

class LinearRegression(LinearModelEstimator):
    _component_class = _LinearRegression

    _default_parameters = {
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
        "n_jobs": None,
        "positive": False
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {
        ProblemType.REGRESSION,
    }