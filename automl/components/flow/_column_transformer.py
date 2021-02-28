from sklearn.compose import ColumnTransformer as _ColumnTransformer
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