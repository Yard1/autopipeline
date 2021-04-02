from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neighbors import KNeighborsTransformer, NeighborhoodComponentsAnalysis
from sklearn.utils.validation import check_is_fitted

from .knn import KNN
from ..transformer import DataType
from ...component import ComponentLevel
from ..transformer import Transformer
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ....problems import ProblemType
from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionDistribution,
)


class NeighborhoodComponentsAnalysisCaching(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        nca_transformer=None,
        knn_transformer=None,
        random_state=None,
    ):
        self.nca_transformer = nca_transformer or NeighborhoodComponentsAnalysis(
            verbose=1
        )
        self.knn_transformer = knn_transformer or KNeighborsTransformer()
        self.random_state = random_state

    def fit(self, X, y):
        self.nca_transformer_ = clone(self.nca_transformer)
        self.nca_transformer_.set_params(random_state=self.random_state)
        Xt = self.nca_transformer_.fit_transform(X, y)
        self.knn_transformer_ = clone(self.knn_transformer)
        self.knn_transformer_.fit(Xt, y=y)
        return self

    def transform(self, X):
        check_is_fitted(self)
        Xt = self.nca_transformer_.transform(X)
        return self.knn_transformer_.transform(Xt)

    def fit_transform(self, X, y, **fit_params):
        self.nca_transformer_ = clone(self.nca_transformer)
        self.nca_transformer_.set_params(random_state=self.random_state)
        Xt = self.nca_transformer_.fit_transform(X, y)
        self.knn_transformer_ = clone(self.knn_transformer)
        return self.knn_transformer_.fit_transform(Xt, y=y)


class KNNTransformer(KNN):
    _component_class = KNeighborsTransformer
    _default_parameters = {
        "mode": "distance",
        "n_neighbors": 40,
        "algorithm": "auto",
        "leaf_size": 30,
        "metric": "minkowski",
        "p": 2,
        "metric_params": None,
        "n_jobs": 1,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.COMMON
    _problem_types = {ProblemType.REGRESSION}

    _default_tuning_grid = {
        "p": IntUniformDistribution(1, 2),
    }


class NCATransformer(KNN):
    _component_class = NeighborhoodComponentsAnalysisCaching
    _default_parameters = {
        "nca_transformer": None,
        "knn_transformer": KNNTransformer()(),
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.COMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}
