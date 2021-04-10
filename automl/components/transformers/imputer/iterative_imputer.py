import pandas as pd
import numpy as np
import gc
from collections import defaultdict
from pandas.api.types import is_categorical_dtype, is_integer_dtype
from sklearn.base import clone
from sklearn.impute._base import (
    _get_mask,
    _most_frequent,
    _check_inputs_dtype,
    FLOAT_DTYPES,
    is_scalar_nan,
)
from sklearn.impute._base import _BaseImputer
from sklearn.utils.validation import check_is_fitted
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

from ..encoder.ordinal_encoder import PandasOrdinalEncoder
from .imputer import Imputer
from ..transformer import Transformer, DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

lightgbm_imputer_config = {
    "n_jobs": 1,
    "n_estimators": 200,
    "class_weight": "balanced",
    "verbose": -1,
    "learning_rate": 0.05,
}


class PandasIterativeImputer(PandasDataFrameTransformerMixin, _BaseImputer):
    def __init__(
        self,
        *,
        missing_values=np.nan,
        regressor=None,
        classifier=None,
        max_iter=10,
        verbose=0,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(missing_values=missing_values, add_indicator=False)
        self.regressor = regressor
        self.classifier = classifier
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _impute(
        self, X_imputed, fit, regressor_fit_params=None, classifier_fit_params=None
    ):
        iter_count = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf

        self.dtypes_ = X_imputed.dtypes
        index = X_imputed.index
        X_imputed.reset_index(drop=True, inplace=True)

        missing_indices = np.where(X_imputed.isna())
        missing_indices_dict = defaultdict(set)
        for idx in range(len(missing_indices[0])):
            missing_indices_dict[X_imputed.columns[missing_indices[1][idx]]].add(
                missing_indices[0][idx]
            )
        range_set = set(range(X_imputed.shape[0]))
        del missing_indices

        max_iter = self.max_iter if fit else 1

        while (
            gamma_new < gamma_old or gamma_newcat < gamma_oldcat
        ) and iter_count < max_iter:
            new_gamma_new = 0
            new_gamma_new_divisor = 0
            new_gamma_newcat = 0
            if iter_count > 1:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat

            for col in self.nan_columns_:
                if self.verbose > 1:
                    print(f"Imputing column: {col}")
                classification = True
                model = self.classifiers_.get(col, None)
                if model is None:
                    classification = False
                    model = self.regressors_[col]

                if fit:
                    X_without_missing = X_imputed.iloc[
                        list(range_set - missing_indices_dict[col])
                    ]
                    X_to_fit = X_without_missing.drop(col, axis=1)
                    y_to_fit = X_without_missing[col]
                    fit_params = (
                        classifier_fit_params
                        if classification
                        else regressor_fit_params
                    )
                    fit_params = fit_params[col] if fit_params else {}
                    model.fit(X_to_fit, y_to_fit, **fit_params)

                missing_indices_list = list(missing_indices_dict[col])
                prediction = model.predict(
                    X_imputed.iloc[missing_indices_list].drop(col, axis=1)
                )
                prediction = (
                    pd.Series(prediction, index=missing_indices_list, name=col)
                    .astype("uint8")
                    .astype(self.dtypes_[col])
                )

                if fit and iter_count != 0:
                    if classification:
                        differences = (
                            X_imputed.iloc[missing_indices_list][col].cat.codes
                            - prediction.cat.codes
                        )
                        new_gamma_newcat += differences.astype(bool).sum()
                    else:
                        new_gamma_new += (
                            prediction - X_imputed.iloc[missing_indices_list][col]
                        ) ** 2
                        new_gamma_new_divisor += (
                            X_imputed.iloc[missing_indices_list][col] ** 2
                        )

                X_imputed.update(prediction)
                X_imputed = X_imputed.astype(self.dtypes_)
                gc.collect()

            if self.classifiers_:
                gamma_newcat = new_gamma_newcat / self.n_catmissing_
            if self.regressors_:
                gamma_new = new_gamma_new / new_gamma_new_divisor

            if self.verbose > 0:
                print(
                    f"Iteration: {iter_count} gamma_new: {gamma_new} gamma_old {gamma_old}, gamma_newcat: {gamma_newcat} gamma_oldcat: {gamma_oldcat}"
                )
            iter_count += 1

        if fit:
            self.iter_count_ = iter_count

        def round_integer_cols(col):
            if col.name in self.integer_cols_:
                return col.round()
            return col

        X_imputed = X_imputed.apply(round_integer_cols).astype(self.dtypes_)

        X_imputed.index = index

        return X_imputed

    def _fit(self, X, y=None, regressor_fit_params=None, classifier_fit_params=None):
        nan_counts = X.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        self.regressors_ = {}
        self.classifiers_ = {}
        self.nan_columns_ = list(nan_counts.index)

        self.integer_cols_ = {
            col for col in X.columns if is_integer_dtype(X.dtypes[col])
        }
        self.categorical_cols_ = {
            col for col in X.columns if is_categorical_dtype(X.dtypes[col])
        }

        self.encoder_ = PandasOrdinalEncoder()

        regressor = self.regressor or LGBMRegressor()
        classifier = self.classifier or LGBMClassifier()

        if regressor == "LGBMRegressor":
            regressor = LGBMRegressor(**lightgbm_imputer_config)
        if classifier == "LGBMClassifier":
            classifier = LGBMClassifier(**lightgbm_imputer_config)

        self.n_catmissing_ = 0
        for col in self.nan_columns_:
            if is_categorical_dtype(X.dtypes[col]):
                self.n_catmissing_ += nan_counts[col]
                self.classifiers_[col] = clone(classifier)
                try:
                    self.classifiers_[col].set_params(random_state=self.random_state)
                except Exception:
                    pass
                try:
                    self.classifiers_[col].set_params(n_jobs=self.n_jobs)
                except Exception:
                    pass
            else:
                self.regressors_[col] = clone(regressor)
                try:
                    self.regressors_[col].set_params(random_state=self.random_state)
                except Exception:
                    pass
                try:
                    self.regressors_[col].set_params(n_jobs=self.n_jobs)
                except Exception:
                    pass

        X_imputed = self.encoder_.fit_transform(X)
        X_imputed = self._impute(
            X_imputed,
            fit=True,
            regressor_fit_params=regressor_fit_params,
            classifier_fit_params=classifier_fit_params,
        )

        return self.encoder_.inverse_transform(X_imputed)

    def fit(self, X, y=None, regressor_fit_params=None, classifier_fit_params=None):
        self._fit(
            X,
            y=y,
            regressor_fit_params=regressor_fit_params,
            classifier_fit_params=classifier_fit_params,
        )
        return self

    def transform(self, X):
        check_is_fitted(self)
        if set(X.columns) != set(self.dtypes_.index):
            raise ValueError(
                f"Column mismatch. Expected {list(self.dtypes_.index)}, got {X.columns}"
            )
        X_imputed = self.encoder_.transform(X)
        X_imputed = self._impute(X_imputed, fit=False)

        return self.encoder_.inverse_transform(X_imputed)

    def fit_transform(
        self, X, y=None, regressor_fit_params=None, classifier_fit_params=None
    ):
        return self._fit(
            X,
            y=y,
            regressor_fit_params=regressor_fit_params,
            classifier_fit_params=classifier_fit_params,
        )


class IterativeImputer(Imputer):
    _component_class = PandasIterativeImputer
    _default_parameters = {
        "regressor": "LGBMRegressor",
        "classifier": "LGBMClassifier",
        "max_iter": 10,
        "verbose": 0,
        "random_state": None,
        "n_jobs": 1,
    }
    _component_level = ComponentLevel.UNCOMMON


# class SimpleCategoricalImputer(Imputer):
#     _component_class = PandasSimpleCategoricalImputer
#     _default_parameters = {
#         "strategy": "most_frequent",
#         "fill_value": "missing_value",
#         "verbose": 0,
#         "copy": True,
#         "add_indicator": False,
#     }
#     _default_tuning_grid = {
#         "strategy": CategoricalDistribution(["most_frequent", "constant"])
#     }
#     _allowed_dtypes = {DataType.CATEGORICAL}
#     _component_level = ComponentLevel.NECESSARY