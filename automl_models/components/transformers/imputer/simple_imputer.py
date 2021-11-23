from pandas.api.types import is_categorical_dtype
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.utils.validation import check_is_fitted

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasSimpleImputer(PandasDataFrameTransformerMixin, _SimpleImputer):
    pass


class PandasSimpleCategoricalImputer(PandasSimpleImputer):
    def _validate_input(self, X, in_fit):
        allowed_strategies = ["most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} "
                " got strategy={1}".format(allowed_strategies, self.strategy)
            )

        if not X.dtypes.apply(is_categorical_dtype).all():
            raise ValueError(
                "Only Pandas DataFrames with all Categorical dtypes are supported."
            )

        return X

    def fit(self, X, y=None):
        X = self._validate_input(X, in_fit=True)
        fill_value = self.fill_value or "missing_value"
        self.statistics_ = {
            col: (
                fill_value
                if self.strategy == "constant" and X[col].isnull().any()
                else X[col].mode()[0]
            )
            for col in X.columns
        }
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=False)

        for col in X.columns:
            if self.statistics_[col] not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories(self.statistics_[col])
        return X.fillna(self.statistics_)

    def get_dtypes(self, Xt, X, y=None):
        return Xt.dtypes.to_dict()
