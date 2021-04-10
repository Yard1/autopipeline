import pandas as pd

from imblearn.over_sampling import (
    SMOTE as _SMOTE,
    SMOTEN as _SMOTEN,
    SMOTENC as _SMOTENC,
)
from sklearn.base import BaseEstimator, clone

from ...compatibility.pandas import *
from .imblearn import ImblearnSampler
from ..transformer import DataType
from ...component import ComponentLevel
from ....problems import ProblemType
from ....search.distributions import IntUniformDistribution


class PandasAutoSMOTE(BaseEstimator):
    """Automatically choose which SMOTE to use based on features"""

    def __init__(
        self, k_neighbors=5, sampling_strategy="auto", random_state=None, n_jobs=None
    ) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self._all_categorical_sampler = _SMOTEN(
            random_state=random_state,
            n_jobs=n_jobs,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        self._all_numeric_sampler = _SMOTE(
            random_state=random_state,
            n_jobs=n_jobs,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        self._mixed_sampler = _SMOTENC(
            categorical_features=[],
            random_state=random_state,
            n_jobs=n_jobs,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        super().__init__()

    def get_index(self, Xt, X, y=None):
        return X.index

    def get_columns(self, Xt, X, y=None):
        return X.columns

    def get_dtypes(self, Xt, X, y=None):
        return X.dtypes.to_dict()

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

    def get_index(self, Xt, X, y=None):
        return None

    def fit_resample(self, X, y, **fit_params):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        try:
            self.columns_ = X.columns
        except:
            self.columns_ = None
        validate_type(X, "X", pd.DataFrame)

        sampler = self._get_sampler(X)

        Xt, yt = sampler.fit_resample(
            categorical_columns_to_int_categories(X).reset_index(drop=True),
            y.reset_index(drop=True),
        )

        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(
                Xt, index=self.get_index(Xt, X), columns=self.get_columns(Xt, X)
            )
        Xt = Xt.astype(self.get_dtypes(Xt, X))

        if not isinstance(yt, pd.Series):
            yt.name = y.name
            yt = yt.astype(y.dtype)

        return Xt, yt


class AutoSMOTE(ImblearnSampler):
    _component_class = PandasAutoSMOTE
    _default_parameters = {
        "k_neighbors": 5,
        "sampling_strategy": "auto",
        "random_state": None,
        "n_jobs": None,
    }
    _component_level = ComponentLevel.UNCOMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}

    _default_tuning_grid = {
        "k_neighbors": IntUniformDistribution(2, 20, log=True, cost_related=False),
    }