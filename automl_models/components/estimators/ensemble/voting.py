from functools import partial
import numpy as np

import ray
import ray.exceptions
from joblib import Parallel

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder  # TODO: consider PandasLabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.fixes import delayed
from sklearn.ensemble import (
    VotingClassifier as _VotingClassifier,
    VotingRegressor as _VotingRegressor,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .utils import (
    call_method,
    ray_put_if_needed,
    should_use_ray,
    ray_call_method,
    fit_estimators,
    get_ray_pg,
    ray_pg_context
)
from ...utils import clone_with_n_jobs


def _get_predictions(parallel, estimators, X, method, pg=None):
    if should_use_ray(parallel):
        estimators = [ray_put_if_needed(est) for est in estimators]
        X_ref = ray_put_if_needed(X)
        predictions = ray.get(
            [
                ray_call_method.options(
                    placement_group=pg, num_cpus=pg.bundle_specs[-1]["CPU"] if pg else 1
                ).remote(est, method, X_ref)
                for est in estimators
            ]
        )
    else:
        predictions = parallel(
            delayed(call_method)(
                est,
                method,
                X,
            )
            for i, est in enumerate(estimators)
        )
    return predictions


# TODO consider accumulation as in _BaseForest to avoid storing all preds
class PandasVotingClassifier(_VotingClassifier):
    _is_ensemble = True

    def fit(self, X, y, sample_weight=None):
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                "Multilabel and multi-output" " classification is not supported."
            )

        if self.voting not in ("soft", "hard"):
            raise ValueError(
                "Voting must be 'soft' or 'hard'; got (voting=%r)" % self.voting
            )

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        y = self.le_.transform(y)

        """Get common fit operations."""
        names, clfs = self._validate_estimators()
        clfs = [clone(est) for est in clfs]

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal"
                "; got %d weights, %d estimators"
                % (len(self.weights), len(self.estimators))
            )

        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(parallel, self.n_jobs, len(clfs))
        with ray_pg_context(pg) as pg:
            self.estimators_ = fit_estimators(
                parallel,
                clfs,
                X,
                y,
                sample_weight,
                partial(clone_with_n_jobs, n_jobs=int(pg.bundle_specs[-1]["CPU"]) if pg else 1),
                pg=pg,
            )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

        return self

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] != "drop"]

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )

        maj = self.le_.inverse_transform(maj)

        return maj

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(parallel, self.n_jobs, len(self.estimators_))
        with ray_pg_context(pg) as pg:
            predictions = _get_predictions(parallel, self.estimators_, X, "predict", pg=pg)

        return np.asarray(predictions).T

    def _collect_probas(self, X):
        """Collect results from clf.predict_proba calls."""
        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(parallel, self.n_jobs, len(self.estimators_))
        with ray_pg_context(pg) as pg:
            predictions = _get_predictions(
                parallel, self.estimators_, X, "predict_proba", pg=pg
            )

        return np.asarray(predictions)

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting."""
        check_is_fitted(self)
        avg = np.average(
            self._collect_probas(X), axis=0, weights=self._weights_not_none
        )
        return avg


class PandasVotingRegressor(_VotingRegressor):
    _is_ensemble = True

    def fit(self, X, y, sample_weight=None):
        """Get common fit operations."""
        names, regs = self._validate_estimators()
        regs = [clone(est) for est in regs]

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal"
                "; got %d weights, %d estimators"
                % (len(self.weights), len(self.estimators))
            )

        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(parallel, self.n_jobs, len(regs))
        with ray_pg_context(pg) as pg:
            self.estimators_ = fit_estimators(
                parallel,
                regs,
                X,
                y,
                sample_weight,
                partial(clone_with_n_jobs, n_jobs=int(pg.bundle_specs[-1]["CPU"]) if pg else 1),
                pg=pg,
            )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

        return self

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] != "drop"]

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1, weights=self._weights_not_none)

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        parallel = Parallel(n_jobs=self.n_jobs)
        pg = get_ray_pg(parallel, self.n_jobs, len(self.estimators_))
        with ray_pg_context(pg) as pg:
            predictions = _get_predictions(parallel, self.estimators_, X, "predict", pg=pg)

        return np.asarray(predictions).T
