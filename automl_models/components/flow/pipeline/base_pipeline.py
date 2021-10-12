from joblib.memory import NotMemorizedFunc
import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline as _ImblearnPipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import is_classifier, clone, is_regressor
from sklearn.pipeline import Pipeline, check_memory
from time import time

from ...utils import validate_type


def _transform_one(
    transformer,
    X,
):
    return transformer.transform(X)


def _inverse_transform_one(
    transformer,
    X,
):
    return transformer.inverse_transform(X)


def _fit(estimator, X, y, *args, **kwargs):
    return estimator.fit(X, y, *args, **kwargs)


def _fit_transform(estimator, X, y, *args, **kwargs):
    return estimator.fit_transform(X, y, *args, **kwargs), estimator


def _fit_resample(estimator, X, y, *args, **kwargs):
    return estimator.fit_resample(X, y, *args, **kwargs), estimator


def _fit_predict(estimator, X, y, *args, **kwargs):
    return estimator.fit_predict(X, y, *args, **kwargs), estimator


def _predict(estimator, X, *args, **kwargs):
    return estimator.predict(X, *args, **kwargs)


def _predict_proba(estimator, X, *args, **kwargs):
    return estimator.predict_proba(X, *args, **kwargs)


def _predict_log_proba(estimator, X, *args, **kwargs):
    return estimator.predict_log_proba(X, *args, **kwargs)


def _decision_function(estimator, X, *args, **kwargs):
    return estimator.decision_function(X, *args, **kwargs)


def _score(estimator, X, y, *args, **kwargs):
    return estimator.score(X, y, *args, **kwargs)


def _score_samples(estimator, X, *args, **kwargs):
    return estimator.score_samples(X, *args, **kwargs)


class BasePipeline(_ImblearnPipeline):
    def __init__(self, steps, *, memory=None, verbose=False, target_pipeline=None):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        assert target_pipeline is None or isinstance(target_pipeline, Pipeline)
        self.target_pipeline = target_pipeline
        self._validate_steps()
        if target_pipeline:
            assert is_regressor(self._final_estimator) or is_classifier(
                self._final_estimator
            )

    def _memory_cache_if_not_pipeline(self, est, func, memory):
        if isinstance(est, BasePipeline):
            return NotMemorizedFunc(func)
        return memory.cache(func)

    @property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return "passthrough" if estimator is None else estimator

    @_final_estimator.setter
    def _final_estimator(self, estimator):
        self.steps[-1] = (self.steps[-1][0], estimator)

    def _transform(self, X, with_final: bool = True):
        X, _ = self._convert_to_df_if_needed(X)

        memory = check_memory(self.memory)
        transform_one_cached = memory.cache(_transform_one)

        Xt = X
        for _, _, transform in self._iter(with_final=with_final):
            f = transform_one_cached
            if isinstance(transform, BasePipeline):
                f = NotMemorizedFunc(transform_one_cached.func)
            Xt = f(transform, Xt)
        return Xt

    def _inverse_transform(self, X):
        X, _ = self._convert_to_df_if_needed(X)

        memory = check_memory(self.memory)
        inverse_transform_one_cached = memory.cache(_inverse_transform_one)

        Xt = X
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            f = inverse_transform_one_cached
            if isinstance(transform, BasePipeline):
                f = NotMemorizedFunc(inverse_transform_one_cached.func)
            Xt = f(transform, Xt)
        return Xt

    def set_params(self, **kwargs):
        # ConfigSpace workaround
        kwargs = {k: (None if v == "!None" else v) for k, v in kwargs.items()}
        if self.target_pipeline:
            self.target_pipeline.set_params(
                **{k: v for k, v in kwargs.items() if k in ("memory", "verbose")}
            )
        else:
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("target_pipeline__")}
        return super().set_params(**kwargs)

    def _convert_to_df_if_needed(self, X, y=None, fit=False):
        if not hasattr(self, "X_columns_"):
            validate_type(X, "X", pd.DataFrame)
        if fit and isinstance(X, pd.DataFrame):
            self.X_columns_ = X.columns
            self.X_dtypes_ = X.dtypes
        elif not isinstance(X, pd.DataFrame):
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
        if self.target_pipeline:
            self.target_pipeline = clone(self.target_pipeline)
            y = self.target_pipeline.fit_transform(y)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        memory = check_memory(self.memory)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                final_estimator_time_start = time()
                fit_cached = self._memory_cache_if_not_pipeline(
                    self._final_estimator, _fit, memory
                )
                self._final_estimator = fit_cached(
                    self._final_estimator, Xt, yt, **fit_params_last_step
                )
                self.final_estimator_fit_time_ = time() - final_estimator_time_start
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        if self.target_pipeline:
            self.target_pipeline = clone(self.target_pipeline)
            y = self.target_pipeline.fit_transform(y)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        memory = check_memory(self.memory)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            final_estimator_time_start = time()
            if hasattr(self._final_estimator, "fit_transform"):
                fit_transform_cached = self._memory_cache_if_not_pipeline(
                    self._final_estimator, _fit_transform, memory
                )
                r, self._final_estimator = fit_transform_cached(
                    self._final_estimator, Xt, yt, **fit_params_last_step
                )
            else:
                fit_cached = self._memory_cache_if_not_pipeline(
                    self._final_estimator, _fit, memory
                )
                self._final_estimator = fit_cached(
                    self._final_estimator, Xt, yt, **fit_params_last_step
                )
                transform_cached = self._memory_cache_if_not_pipeline(
                    self._final_estimator, _transform_one, memory
                )
                r = transform_cached(self._final_estimator, Xt)
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
            return r

    def fit_resample(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        if self.target_pipeline:
            self.target_pipeline = clone(self.target_pipeline)
            y = self.target_pipeline.fit_transform(y)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        memory = check_memory(self.memory)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            final_estimator_time_start = time()
            if hasattr(self._final_estimator, "fit_resample"):
                fit_resample_cached = self._memory_cache_if_not_pipeline(
                    self._final_estimator, _fit_resample, memory
                )
                r, self._final_estimator = fit_resample_cached(
                    self._final_estimator, Xt, yt, **fit_params_last_step
                )
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
            return r

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_predict(self, X, y=None, **fit_params):
        X, y = self._convert_to_df_if_needed(X, y, fit=True)
        if self.target_pipeline:
            self.target_pipeline = clone(self.target_pipeline)
            y = self.target_pipeline.fit_transform(y)
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        memory = check_memory(self.memory)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            final_estimator_time_start = time()
            fit_predict_cached = self._memory_cache_if_not_pipeline(
                self._final_estimator, _fit_predict, memory
            )
            y_pred, self._final_estimator = fit_predict_cached(
                self._final_estimator, Xt, yt, **fit_params_last_step
            )
            self.final_estimator_fit_time_ = time() - final_estimator_time_start
        if self.target_pipeline:
            y_pred = self.target_pipeline.inverse_transform(y_pred)
        return y_pred

    @if_delegate_has_method(delegate="_final_estimator")
    def predict(self, X, **predict_params):
        Xt = self._transform(X, with_final=False)
        memory = check_memory(self.memory)
        predict_cached = self._memory_cache_if_not_pipeline(
            self._final_estimator, _predict, memory
        )
        y_pred = predict_cached(self._final_estimator, Xt, **predict_params)
        if self.target_pipeline:
            y_pred = self.target_pipeline.inverse_transform(y_pred)
        return y_pred

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_proba(self, X, **predict_params):
        Xt = self._transform(X, with_final=False)
        memory = check_memory(self.memory)
        predict_proba_cached = self._memory_cache_if_not_pipeline(
            self._final_estimator, _predict_proba, memory
        )
        return predict_proba_cached(self._final_estimator, Xt, **predict_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def decision_function(self, X, **decision_params):
        Xt = self._transform(X, with_final=False)
        memory = check_memory(self.memory)
        decision_function_cached = self._memory_cache_if_not_pipeline(
            self._final_estimator, _decision_function, memory
        )
        return decision_function_cached(self._final_estimator, Xt, **decision_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def score_samples(self, X, **score_params):
        Xt = self._transform(X, with_final=False)
        memory = check_memory(self.memory)
        score_samples_cached = memory.cache(_score_samples)
        return score_samples_cached(self._final_estimator, Xt, **score_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_log_proba(self, X, **predict_params):
        Xt = self._transform(X, with_final=False)
        memory = check_memory(self.memory)
        predict_log_proba_cached = self._memory_cache_if_not_pipeline(
            self._final_estimator, _predict_log_proba, memory
        )
        return predict_log_proba_cached(self._final_estimator, Xt, **predict_params)

    @if_delegate_has_method(delegate="_final_estimator")
    def score(self, X, y=None, **score_params):
        if is_classifier(self._final_estimator):
            from sklearn.metrics import accuracy_score

            return accuracy_score(y, self.predict(X), **score_params)
        elif is_regressor:
            from sklearn.metrics import r2_score

            y_pred = self.predict(X)
            return r2_score(y, y_pred, **score_params)
        else:
            Xt = self._transform(X, with_final=False)
            memory = check_memory(self.memory)
            score_cached = memory.cache(_score)
            return score_cached(self._final_estimator, Xt, **score_params)
