import pandas as pd

from sklearn.preprocessing import LabelEncoder as _LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ...compatibility.pandas import PandasSeriesTransformerMixin


# TODO look into .codes
class PandasLabelEncoder(PandasSeriesTransformerMixin, _LabelEncoder):
    def get_dtype(self, r, X, y=None):
        return pd.CategoricalDtype(self.indices_.values())

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()
        if isinstance(y, pd.Series):
            y = y.astype("category")
            self.classes_ = y.cat.categories.to_numpy()
            self.indices_ = {cls: index for index, cls in enumerate(self.classes_)}
        else:
            return super().fit(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self.fit(y)
        return self.transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self)

        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()
        if isinstance(y, pd.Series):
            yt = y.astype(pd.CategoricalDtype(categories=self.classes_))
            if yt.isnull().any():
                diff = set(y.unique()) - set(self.classes_)
                raise ValueError(f"y contains previously unseen labels: {list(diff)}")
            yt = yt.cat.rename_categories(self.indices_)
            yt.index = y.index
            yt.name = y.name
            y = yt
        else:
            return super().fit(y)
        return y

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self)

        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()
        if isinstance(y, pd.Series):
            y = y.astype("category")
            diff = set(y.cat.categories) - set(range(len(self.classes_)))
            if diff:
                raise ValueError(f"y contains previously unseen labels: {list(diff)}")
            y = y.cat.rename_categories({v: k for k, v in self.indices_.items()})
        else:
            return super().fit(y)
        return y
