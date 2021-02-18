import numpy as np
import pandas as pd
from pandas.core.indexes import category

from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder

from ..component import Component
from ..compatibility.pandas import PandasDataFrameTransformerMixin


class PandasOneHotEncoder(PandasDataFrameTransformerMixin, _OneHotEncoder):
    def get_columns(self, Xt, X, y=None):
        columns = []
        for column, categories in zip(self.columns_, self.categories_):
            columns.extend([f"{column}_{category}" for category in categories])
        return columns

    def get_dtypes(self, Xt, X, y=None):
        return pd.CategoricalDtype([0, 1])



class OneHotEncoder(Component):
    _component_class = PandasOneHotEncoder
    _default_parameters = {
        "categories": "auto",
        "drop": "if_binary",
        "sparse": False,
        "dtype": np.int,
    }
