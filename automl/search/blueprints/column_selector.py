import pandas as pd
from sklearn.compose import make_column_selector as _make_column_selector

class make_column_selector(_make_column_selector):
    def __init__(self, condition=None, *, dtype_include=None,
                 dtype_exclude=None, negate_condition=False):
        self.condition = condition
        self.negate_condition = negate_condition
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, df):
        if not hasattr(df, 'iloc'):
            raise ValueError("make_column_selector can only be applied to "
                             "pandas dataframes")
        df_row = df.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(include=self.dtype_include,
                                          exclude=self.dtype_exclude)
        cols = df_row.columns
        if self.condition is not None:
            if self.negate_condition:
                cols = [col for col in cols if not self.condition(df_row[col])]
            else:
                cols = [col for col in cols if self.condition(df_row[col])]
        return list(cols)
