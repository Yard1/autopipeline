import pandas as pd

from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasOneHotEncoder(PandasDataFrameTransformerMixin, _OneHotEncoder):
    def get_columns(self, Xt, X, y=None):
        columns = []
        for column, categories in zip(self.columns_, self.categories_):
            if self.drop == "first" or (
                self.drop == "if_binary" and len(categories) == 2
            ):
                categories = categories[1:]
            columns.extend([f"{column}_{category}" for category in categories])
        return columns

    def get_dtypes(self, Xt, X, y=None):
        return pd.CategoricalDtype([0, 1])

    def _validate_keywords(self):
        self._infrequent_enabled = False
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', "
                "got {0}.".format(self.handle_unknown)
            )
            raise ValueError(msg)

    def fit(self, X, y=None):
        # we assume we only have categorical columns here
        X = X.apply(lambda col: col.cat.codes)
        return super().fit(X, y=y)

    def transform(self, X):
        # we assume we only have categorical columns here
        X = X.apply(lambda col: col.cat.codes)
        return super().transform(X)
