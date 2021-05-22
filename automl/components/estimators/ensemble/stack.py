from .ensemble import Ensemble
from ....problems import ProblemType

from automl_models.components.estimators.ensemble.stack import (
    PandasStackingClassifier,
    PandasStackingRegressor,
)

import logging

logger = logging.getLogger(__name__)


class StackingClassifier(Ensemble):
    _component_class = PandasStackingClassifier

    _default_parameters = {}

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class StackingRegressor(Ensemble):
    _component_class = PandasStackingRegressor

    _default_parameters = {}

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.REGRESSION}
