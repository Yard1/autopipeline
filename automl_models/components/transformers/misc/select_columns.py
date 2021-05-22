from sklearn.base import BaseEstimator, TransformerMixin


class PandasSelectColumns(TransformerMixin, BaseEstimator):
    def __init__(self, columns_to_select, *, drop: bool = False) -> None:
        self.columns_to_select = columns_to_select
        self.drop = drop
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.drop:
            return X.drop(self.columns_to_select, axis=1)
        return X[self.columns_to_select] 
