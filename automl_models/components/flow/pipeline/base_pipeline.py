import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline as _ImblearnPipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import is_classifier
from time import time

from ...utils import validate_type


class BasePipeline(_ImblearnPipeline):
    def set_params(self, **kwargs):
        # ConfigSpace workaround
        kwargs = {k: (None if v == "!None" else v) for k, v in kwargs.items()}
        return super().set_params(**kwargs)

    def _convert_to_df_if_needed(self, X, y=None, fit=False):
        if not hasattr(self, "X_columns_"):
            validate_type(X, "X", pd.DataFrame)
        if fit and isinstance(X, pd.DataFrame):
            self.X_columns_ = X.columns
            self.X_dtypes_ = X.dtypes
        else:
            X = pd.DataFrame(X, columns=self.X_columns_)
            X = X.astype(self.X_dtypes_)
        if y is not None:
            if fit:
                if isinstance(y, pd.Series):
                    self.y_name_ = y.name
                    self.y_dtype_ = y.dtype
                else:
                    self.y_name_ = "target"
                    if is_classifier(self._final_estimator):
                        y = y.astype(int)
                        self.y_dtype_ = "category"
                    else:
                        self.y_dtype_ = np.float32  # TODO make dynamic
                    y = pd.Series(y, name=self.y_name_)
                    y = y.astype(self.y_dtype_)
            else:
                y = pd.Series(y, name=self.y_name_)
                y = y.astype(self.y_dtype_)
        return X, y

    def fit(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                final_estimator_time_start = time()
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)
                self.final_estimator_fit_time_ = time() - final_estimator_time_start
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            final_estimator_time_start = time()
            if hasattr(last_step, "fit_transform"):
                r = last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                r = last_step.fit(Xt, yt, **fit_params_last_step).transform(Xt)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
            return r

    def fit_resample(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            final_estimator_time_start = time()
            if hasattr(last_step, "fit_resample"):
                r = last_step.fit_resample(Xt, yt, **fit_params_last_step)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
            return r

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_predict(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            final_estimator_time_start = time()
            y_pred = self.steps[-1][-1].fit_predict(Xt, yt, **fit_params_last_step)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
        return y_pred

    @if_delegate_has_method(delegate="_final_estimator")
    def predict(self, X, **predict_params):
        X, _ = self._convert_to_df_if_needed(X)
        return super().predict(X=X, **predict_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_proba(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().predict_proba(X=X)

    @if_delegate_has_method(delegate="_final_estimator")
    def decision_function(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().decision_function(X=X)

    @if_delegate_has_method(delegate="_final_estimator")
    def score_samples(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().score_samples(X=X)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_log_proba(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super().predict_log_proba(X=X)

    def _transform(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super()._transform(X)

    def _inverse_transform(self, X):
        X, _ = self._convert_to_df_if_needed(X)
        return super()._inverse_transform(X)

    @if_delegate_has_method(delegate="_final_estimator")
    def score(self, X, y=None, sample_weight=None):
        X, y = self._convert_to_df_if_needed(X, y)
        return super().score(X=X, y=y, sample_weight=sample_weight)
