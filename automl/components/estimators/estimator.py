from enum import Enum

from ..component import Component


class ModelType(Enum):
    KNN = "knn"
    LINEAR = "linear"
    SVM = "svm"
    RANDOM_FOREST = "rf"
    EXTRA_TREES = "et"
    ENSEMBLE = "ensemble"
    DECISION_TREE = "dt"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    SPECIAL = "special"
    OTHER = "other"

    def is_tree(self):
        return self in {
            self.RANDOM_FOREST,
            self.EXTRA_TREES,
            self.DECISION_TREE,
            self.CATBOOST,
            self.XGBOOST,
            self.LIGHTGBM,
        }


class Estimator(Component):
    _model_type = None
    _problem_types = {}

    def __init__(self, **parameters) -> None:
        if not isinstance(self._model_type, ModelType):
            raise TypeError(
                f"_model_type must be of type ModelType, got '{type(self._model_type)}'"
            )
        if len(self._problem_types) != 1:
            raise ValueError(
                f"_problem_types must have length 1, got {len(self._problem_types)}"
            )
        super().__init__(**parameters)

