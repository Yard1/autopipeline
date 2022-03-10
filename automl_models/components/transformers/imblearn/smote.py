import pandas as pd

from imblearn.over_sampling import (
    SMOTE as _SMOTE,
    SMOTEN as _SMOTEN,
    SMOTENC as _SMOTENC,
)
from sklearn.base import BaseEstimator, clone

from ...compatibility.pandas import PandasDataFrameTransformerMixin
from ..encoder.ordinal_encoder import PandasOrdinalEncoder
from ..transformer import DataType
from ...utils import validate_type

# TODO consider caching for KNN inside SMOTE?


class SMOTENJobsMixin:
    def _validate_estimator(self):
        super()._validate_estimator()
        try:
            self.nn_k_.set_params(n_jobs=self.n_jobs)
        except ValueError:
            pass


class SMOTEN(SMOTENJobsMixin, _SMOTEN):
    pass


class SMOTE(SMOTENJobsMixin, _SMOTE):
    pass


class SMOTENC(SMOTENJobsMixin, _SMOTENC):
    pass


class _PandasAutoSMOTE(BaseEstimator):
    """Automatically choose which SMOTE to use based on features"""

    def __init__(
        self, k_neighbors=5, sampling_strategy="auto", random_state=None, n_jobs=None
    ) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self._all_categorical_sampler = SMOTEN(
            random_state=random_state,
            n_jobs=n_jobs,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        self._all_numeric_sampler = SMOTE(
            random_state=random_state,
            n_jobs=n_jobs,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        self._mixed_sampler = SMOTENC(
            categorical_features=[],
            random_state=random_state,
            n_jobs=n_jobs,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        super().__init__()

    def set_params(self, **params):
        self._all_categorical_sampler.set_params(**params)
        self._all_numeric_sampler.set_params(**params)
        self._mixed_sampler.set_params(**params)
        return super().set_params(**params)

    def _get_sampler(self, X):
        categorical_columns = X.dtypes.apply(lambda x: DataType.is_categorical(x))
        categorical_columns_mask = categorical_columns.astype(bool).to_numpy()
        num_categorical_columns = categorical_columns_mask.sum()
        if num_categorical_columns == 0:
            return clone(self._all_numeric_sampler)
        if num_categorical_columns == len(categorical_columns_mask):
            return clone(self._all_categorical_sampler)

        mixed_sampler = clone(self._mixed_sampler)
        mixed_sampler.set_params(categorical_features=categorical_columns_mask)
        return mixed_sampler

    def fit_resample(self, X, y, **fit_params):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        try:
            self.columns_ = X.columns
        except Exception:
            self.columns_ = None
        validate_type(X, "X", pd.DataFrame)

        sampler = self._get_sampler(X)

        encoder = PandasOrdinalEncoder()

        Xt, yt = sampler.fit_resample(
            encoder.fit_transform(X).reset_index(drop=True),
            y.reset_index(drop=True),
        )

        Xt = encoder.inverse_transform(Xt)

        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(
                Xt, index=self.get_index(Xt, X), columns=self.get_columns(Xt, X)
            )
        Xt = Xt.astype(self.get_dtypes(Xt, X))

        if not isinstance(yt, pd.Series):
            yt.name = y.name
            yt = yt.astype(y.dtype)

        return Xt, yt


class PandasAutoSMOTE(PandasDataFrameTransformerMixin, _PandasAutoSMOTE):
    def get_index(self, Xt, X, y=None):
        return Xt.index
