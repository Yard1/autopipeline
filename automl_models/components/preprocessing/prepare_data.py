from typing import Optional, List, Dict

import pandas as pd
import numpy as np
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype,
)

from fastai.tabular.core import df_shrink, add_datepart
from pandas.core.dtypes.common import is_bool_dtype, is_categorical_dtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import warnings
from ..utils import validate_type


def _is_id_column(col):
    return (
        is_integer_dtype(col.dtype)
        and not col.isnull().any()
        and (col == sorted(col)).sum() == len(col)
        and len(col) == len(col.unique())
    )


def replace_inf_in_col(col):
    try:
        return col.replace([np.inf, -np.inf], None)  # TODO: ensure this actually works
    except Exception:
        return col


def clean_df(df):
    if isinstance(df, pd.DataFrame):
        df.columns = [str(col) for col in df.columns]
    else:
        df.name = str(df.name)
    df = df.apply(replace_inf_in_col)
    return df


class PrepareDataFrame(TransformerMixin, BaseEstimator):
    _datetime_dtype = "datetime64[ns]"

    def __init__(
        self,
        allowed_dtypes: Optional[List[type]] = None,
        find_id_column: bool = True,
        float_dtype: type = np.float32,
        int_dtype: Optional[type] = None,
        ordinal_columns: Optional[Dict[str, list]] = None,
        copy_X: bool = True,
    ) -> None:
        if allowed_dtypes is not None and not allowed_dtypes:
            raise ValueError("allowed_dtypes cannot be empty")
        if not is_float_dtype(float_dtype):
            raise TypeError(
                f"Expected float_dtype to be a float dtype, got {type(float_dtype)}"
            )
        if int_dtype is not None and not is_integer_dtype(int_dtype):
            raise TypeError(
                f"Expected int_dtype to be an integer dtype or None, got {type(int_dtype)}"
            )
        validate_type(ordinal_columns, "ordinal_columns", (dict, None))

        self.allowed_dtypes = allowed_dtypes
        self.find_id_column = find_id_column
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.copy_X = copy_X
        self.ordinal_columns = ordinal_columns or {}

    def _set_index_to_id_column(self, X):
        possible_id_columns = X.apply(_is_id_column)
        possible_id_columns = possible_id_columns[possible_id_columns]
        if len(possible_id_columns) > 1:
            warnings.warn(
                f"{len(possible_id_columns)} possible ID columns found ({list(possible_id_columns.index)}). Will not set any as index."
            )
        elif len(possible_id_columns) == 1:
            warnings.warn(
                f"Setting ID column '{possible_id_columns.index[0]}' as index."
            )
            self.id_column_ = possible_id_columns.index[0]
            return X.set_index(self.id_column_)
        return X

    def _infer_dtypes(self, col):
        if is_float_dtype(col.dtype):
            return col.astype(self.float_dtype)

        if is_datetime64_any_dtype(col.dtype):
            return col.astype(self._datetime_dtype)

        if is_integer_dtype(col.dtype) and self.int_dtype is not None:
            return col.astype(self.int_dtype)

        if is_categorical_dtype(col.dtype):
            if col.dtype.ordered:
                col = col.copy()
                return col.cat.codes.replace(-1, None)
            return col

        col_unqiue = col.unique()
        if is_bool_dtype(col.dtype) or (
            not is_numeric_dtype(col.dtype)
            or is_integer_dtype(col.dtype)
            and len(col_unqiue) <= 20
        ):
            try:
                col = pd.to_datetime(
                    col, infer_datetime_format=True, utc=False, errors="raise"
                )
                col = col.astype(self._datetime_dtype)
            except Exception:
                pass
            if col.name in self.ordinal_columns:
                if set(col_unqiue) != set(self.ordinal_columns[col.name]):
                    raise ValueError(
                        f"Ordered values for column '{col.name}' are mismatched. Got {self.ordinal_columns[col.name]}, actual categories {col_unqiue}."
                    )
                return col.astype(
                    pd.CategoricalDtype(self.ordinal_columns[col.name], ordered=True)
                ).cat.codes.replace(-1, None)
            try:
                return col.astype(pd.CategoricalDtype(col_unqiue))
            except Exception:
                return col.astype("category")

        return col

    def _convert_dtypes(self, col):
        if col.name in self.datetime_columns_:
            col = pd.to_datetime(
                col, infer_datetime_format=True, utc=False, errors="raise"
            )
        return col.astype(self.final_dtypes_[col.name])

    def _drop_0_variance(self, X):
        cols_to_drop = []
        for column in X.columns:
            if np.all(X[column] == X[column].iloc[0]):
                cols_to_drop.append(column)
        return X.drop(cols_to_drop, axis=1)

    def fit(self, X, y=None):
        self.fit_transform(X, y=y)

        return self

    def transform(self, X):
        check_is_fitted(self)

        if self.copy_X:
            X = X.copy()

        was_series = isinstance(X, pd.Series)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.infer_objects()

        X = clean_df(X)

        if self.id_column_ is not None:
            X[self.id_column_] = X[self.id_column_].astype(int)
            X.set_index(self.id_column_, inplace=True)

        X = X[self.final_columns_]

        X = X.apply(self._convert_dtypes)

        for datetime_column in self.datetime_columns_:
            add_datepart(X, datetime_column, prefix=f"{datetime_column}_")

        if was_series:
            X = X.squeeze()

        return X

    def fit_transform(self, X, y=None):
        X = X.copy()

        was_series = isinstance(X, pd.Series)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.infer_objects()

        X = clean_df(X)

        X.dropna(axis=0, how="all", inplace=True)
        X.dropna(axis=1, how="all", inplace=True)

        self.id_column_ = None
        if X.shape[1] > 1 and self.find_id_column:
            X = self._set_index_to_id_column(X)

        X = df_shrink(X, obj2cat=False)
        X = X.apply(self._infer_dtypes)
        X = self._drop_0_variance(X)

        self.final_columns_ = X.columns
        self.final_dtypes_ = X.dtypes
        self.datetime_columns_ = self.final_dtypes_.apply(is_datetime64_any_dtype)
        self.datetime_columns_ = set(
            self.datetime_columns_[self.datetime_columns_].index
        )
        for datetime_column in self.datetime_columns_:
            add_datepart(X, datetime_column, prefix=f"{datetime_column}_")

        if was_series:
            X = X.squeeze()

        return X
