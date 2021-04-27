from sklearn.compose import (
    ColumnTransformer as _ColumnTransformer,
    make_column_selector as _make_column_selector,
)
import numpy as np
import pandas as pd
from scipy import sparse


class PandasColumnTransformer(_ColumnTransformer):
    def _hstack(self, Xs):
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
        """
        Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
        try:
            if all(isinstance(X, (pd.DataFrame, pd.Series)) for X in Xs):
                return pd.concat(Xs, axis=1)
        except:
            pass
        return np.hstack(Xs)


class make_column_selector(_make_column_selector):
    def __init__(
        self,
        condition=None,
        *,
        dtype_include=None,
        dtype_exclude=None,
        negate_condition=False
    ):
        self.condition = condition
        self.negate_condition = negate_condition
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, df):
        if not hasattr(df, "iloc"):
            raise ValueError(
                "make_column_selector can only be applied to " "pandas dataframes"
            )
        df_row = df.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(
                include=self.dtype_include, exclude=self.dtype_exclude
            )
        cols = df_row.columns
        if self.condition is not None:
            if self.negate_condition:
                cols = [col for col in cols if not self.condition(df_row[col])]
            else:
                cols = [col for col in cols if self.condition(df_row[col])]
        return list(cols)
