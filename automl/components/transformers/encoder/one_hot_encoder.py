import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder

from ..transformer import Transformer, DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasOneHotEncoder(PandasDataFrameTransformerMixin, _OneHotEncoder):
    def get_columns(self, Xt, X, y=None):
        columns = []
        for column, categories in zip(self.columns_, self.categories_):
            if self.drop == "first" or (self.drop == "if_binary" and len(categories) == 2):
                categories = categories[1:]
            columns.extend([f"{column}_{category}" for category in categories])
        return columns

    def get_dtypes(self, Xt, X, y=None):
        return pd.CategoricalDtype([0, 1])



class OneHotEncoder(Transformer):
    _component_class = PandasOneHotEncoder
    _default_parameters = {
        "categories": "auto",
        "drop": "if_binary",
        "sparse": False,
        "dtype": np.int,
        "handle_unknown": "error",
    }
    _allowed_dtypes = {
        DataType.CATEGORICAL
    }
    _component_level = ComponentLevel.NECESSARY
