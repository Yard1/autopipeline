from .feature_selector import FeatureSelector
from ...component import ComponentLevel
from ....problems import ProblemType

from automl_models.components.transformers.feature_selector.boruta import BorutaSHAP


class BorutaSHAPClassification(FeatureSelector):
    _component_class = BorutaSHAP
    _default_parameters = {
        "estimator": "LGBMClassifier",
        "n_estimators": "auto",
        "perc": 100,
        "alpha": 0.05,
        "two_step": True,
        "max_iter": 100,
        "random_state": 0,
        "verbose": 0,
        "early_stopping": True,
        "n_iter_no_change": 10,
    }
    _component_level = ComponentLevel.RARE
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class BorutaSHAPRegression(FeatureSelector):
    _component_class = BorutaSHAP
    _default_parameters = {
        "estimator": "LGBMRegressor",
        "n_estimators": "auto",
        "perc": 100,
        "alpha": 0.05,
        "two_step": True,
        "max_iter": 100,
        "random_state": 0,
        "verbose": 0,
        "early_stopping": True,
        "n_iter_no_change": 10,
    }
    _component_level = ComponentLevel.RARE
    _problem_types = {ProblemType.REGRESSION}
