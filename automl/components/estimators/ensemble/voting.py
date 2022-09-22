from .ensemble import Ensemble
from ....problems import ProblemType

from automl_models.components.estimators.ensemble.voting import (
    PandasVotingClassifier,
    PandasVotingRegressor,
    PandasGreedyVotingClassifier,
    PandasGreedyVotingRegressor,
)


class VotingClassifier(Ensemble):
    _component_class = PandasVotingClassifier

    _default_parameters = {}

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class VotingRegressor(Ensemble):
    _component_class = PandasVotingRegressor

    _default_parameters = {}

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.REGRESSION}


class GreedyVotingClassifier(Ensemble):
    _component_class = PandasGreedyVotingClassifier

    _default_parameters = {
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


class GreedyVotingRegressor(Ensemble):
    _component_class = PandasGreedyVotingRegressor

    _default_parameters = {
    }

    _default_tuning_grid = {}
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.REGRESSION}
