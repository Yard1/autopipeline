from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neighbors import KNeighborsTransformer, NeighborhoodComponentsAnalysis
from sklearn.utils.validation import check_is_fitted


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
