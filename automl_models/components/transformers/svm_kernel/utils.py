from ...transformers.utils import categorical_column_to_int_categories


class GammaMixin:
    def fit(self, X, y=None, **kwargs):
        n_features = X.shape[1]
        self._gamma = self.gamma
        if self.gamma == "auto":
            self.gamma = 1.0 / n_features
        elif self.gamma == "scale":
            X = X.apply(categorical_column_to_int_categories)
            if hasattr(X, "values"):
                X = X.values
            X_var = X.var()
            self.gamma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
        r = super().fit(X, y=y, **kwargs)
        new_gamma = self.gamma
        self.gamma = self._gamma
        self._gamma = new_gamma
        return r

    def transform(self, X, **kwargs):
        old_gamma = self.gamma
        self.gamma = self._gamma
        r = super().transform(X, **kwargs)
        self.gamma = old_gamma
        return r
