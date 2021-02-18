from copy import deepcopy
from sklearn.compose import ColumnTransformer as _ColumnTransformer

from .transformer import Transformer
from ..uitls import get_step_choice_grid

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.utils.validation import check_array

class PandasColumnTransformer(_ColumnTransformer):
    def _hstack(self, Xs):
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
        """
        if self.sparse_output_:
            try:
                # since all columns should be numeric before stacking them
                # in a sparse matrix, `check_array` is used for the
                # dtype conversion if necessary.
                converted_Xs = [check_array(X,
                                            accept_sparse=True,
                                            force_all_finite=False)
                                for X in Xs]
            except ValueError:
                raise ValueError("For a sparse output, all columns should"
                                 " be a numeric or convertible to a numeric.")

            return sparse.hstack(converted_Xs).tocsr()
        else:
            Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
            try:
                if all(isinstance(X, (pd.DataFrame, pd.Series)) for X in Xs):
                    return pd.concat(Xs, axis=1)
            except:
                pass
            return np.hstack(Xs)

class ColumnTransformer(Transformer):
    _component_class = PandasColumnTransformer
    _default_parameters = {
        "remainder": "passthrough",
        "sparse_threshold": 0,
        "n_jobs": None,
        "transformer_weights": None,
        "verbose": False,
    }

    def __call__(self):
        params = deepcopy(self.final_parameters)
        transformers = [(name, transformer(), columns) for name, transformer, columns in params["transformers"]]
        params["transformers"] = transformers

        return self._component_class(**params)

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = super().get_tuning_grid(use_extended=use_extended)
        transformer_grids = {name:get_step_choice_grid(transformer) for name, transformer, columns in self.final_parameters["transformers"]}
        return {**transformer_grids, **default_grid}