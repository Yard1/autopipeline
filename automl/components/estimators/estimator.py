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
    def __init__(self, **parameters) -> None:
        super().__init__(**parameters)

