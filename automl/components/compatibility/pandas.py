from scipy.sparse import issparse
import pandas as pd
from pandas.api.types import is_categorical_dtype
from ...utils import validate_type

def col_categorical_to_int(column):
    if is_categorical_dtype(column.dtype):
        try:
            return column.astype(int)
        except ValueError:
            return column
    return column


def categorical_to_int(df):
    return df.apply(col_categorical_to_int, axis=1)


class PandasSeriesTransformerMixin:
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            X = X.squeeze(axis=1)
        try:
            self.name_ = X.name
        except:
            self.name_ = None
        validate_type(X, "X", pd.Series)
        try:
            return super().fit(X.to_numpy(), y=y)
        except TypeError:
            return super().fit(X.to_numpy())

    def transform(self, X):
        if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            X = X.squeeze(axis=1)
        validate_type(X, "X", pd.Series)
        Xt = super().transform(col_categorical_to_int(X).to_numpy())
        if not isinstance(Xt, pd.Series):
            if issparse(Xt):
                Xt_s = pd.Series.sparse.from_coo(Xt)
                Xt_s.index = self.get_index(Xt, X)
                Xt_s.name = self.get_name(Xt, X)
                Xt = Xt_s
            else:
                Xt = pd.Series(
                    Xt,
                    index=self.get_index(Xt, X),
                    name=self.get_name(Xt, X),
                )
            Xt = Xt.infer_objects()
            Xt = Xt.astype(self.get_dtype(Xt, X))
        return Xt

    def get_index(self, Xt, X, y=None):
        return X.index

    def get_name(self, Xt, X, y=None):
        return X.name

    def get_dtype(self, Xt, X, y=None):
        return X.dtype


class PandasDataFrameTransformerMixin:
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        try:
            self.columns_ = X.columns
        except:
            self.columns_ = None
        validate_type(X, "X", pd.DataFrame)
        try:
            return super().fit(categorical_to_int(X).to_numpy(), y=y)
        except TypeError:
            return super().fit(categorical_to_int(X).to_numpy())

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, columns=self.columns_)
        validate_type(X, "X", pd.DataFrame)
        if not sorted(self.columns_) == sorted(X.columns):
            raise ValueError(
                f"Column mismatch. Expected {self.columns_}, got {X.columns}"
            )
        X = X[self.columns_]
        Xt = super().transform(categorical_to_int(X).to_numpy())

        if not isinstance(Xt, pd.DataFrame):
            if issparse(Xt):
                Xt = pd.DataFrame.sparse.from_spmatrix(
                    Xt, index=self.get_index(Xt, X), columns=self.get_columns(Xt, X)
                )
            else:
                Xt = pd.DataFrame(
                    Xt, index=self.get_index(Xt, X), columns=self.get_columns(Xt, X)
                )
            Xt = Xt.infer_objects()
            Xt = Xt.astype(self.get_dtypes(Xt, X))
        return Xt

    def get_index(self, Xt, X, y=None):
        return X.index

    def get_columns(self, Xt, X, y=None):
        return X.columns

    def get_dtypes(self, Xt, X, y=None):
        return X.dtypes.to_dict()
