import numpy as np

import ray.exceptions
from joblib import Parallel

from sklearn.preprocessing import LabelEncoder  # TODO: consider PandasLabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.fixes import delayed
from sklearn.ensemble import (
    VotingClassifier as _VotingClassifier,
    VotingRegressor as _VotingRegressor,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .utils import fit_single_estimator_if_not_fitted, call_method
from ...utils import clone_with_n_jobs_1


def _collect_predictions(obj, X, method):
    if hasattr(obj, "_saved_test_predictions") and obj._saved_test_predictions:
        saved_predictions = obj._saved_test_predictions
    else:
        saved_predictions = [None] * len(obj.estimators_)

    assert len(saved_predictions) == len(obj.estimators_)

    predictions = Parallel(n_jobs=obj.n_jobs, verbose=int(bool(obj.verbose)))(
        delayed(call_method)(
            est,
            method,
            X,
        )
        for i, est in enumerate(obj.estimators_)
        if saved_predictions[i] is None or method not in saved_predictions[i]
    )
    predictions = list(predictions)

    for i in range(len(saved_predictions)):
        if saved_predictions[i] is None or method not in saved_predictions[i]:
            saved_predictions[i] = predictions.pop(0)
        else:
            saved_predictions[i] = saved_predictions[i][method]

    return saved_predictions


# TODO consider accumulation as in _BaseForest to avoid storing all preds
class PandasVotingClassifier(_VotingClassifier):
    def fit(self, X, y, sample_weight=None, refit_estimators=True):
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

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal"
                "; got %d weights, %d estimators"
                % (len(self.weights), len(self.estimators))
            )

        try:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    clf,
                    X,
                    y,
                    sample_weight=sample_weight,
                    message_clsname="Voting",
                    message=self._log_message(names[idx], idx + 1, len(clfs)),
                    force_refit=refit_estimators,
                )
                for idx, clf in enumerate(clfs)
                if clf != "drop"
            )
        except (ray.exceptions.RayError):
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    clf,
                    X,
                    y,
                    sample_weight=sample_weight,
                    message_clsname="Voting",
                    message=self._log_message(names[idx], idx + 1, len(clfs)),
                    cloning_function=clone_with_n_jobs_1,
                    force_refit=refit_estimators,
                )
                for idx, clf in enumerate(clfs)
                if clf != "drop"
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
        predictions = _collect_predictions(self, X, "predict")

        return np.asarray(predictions).T

    def _collect_probas(self, X):
        """Collect results from clf.predict_proba calls."""
        predictions = _collect_predictions(self, X, "predict_proba")

        return np.asarray(predictions)

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting."""
        check_is_fitted(self)
        avg = np.average(
            self._collect_probas(X), axis=0, weights=self._weights_not_none
        )
        return avg


class PandasVotingRegressor(_VotingRegressor):
    def fit(self, X, y, sample_weight=None, refit_estimators=True):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal"
                "; got %d weights, %d estimators"
                % (len(self.weights), len(self.estimators))
            )

        try:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    clf,
                    X,
                    y,
                    sample_weight=sample_weight,
                    message_clsname="Voting",
                    message=self._log_message(names[idx], idx + 1, len(clfs)),
                    force_refit=refit_estimators,
                )
                for idx, clf in enumerate(clfs)
                if clf != "drop"
            )
        except (ray.exceptions.RayError):
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_single_estimator_if_not_fitted)(
                    clf,
                    X,
                    y,
                    sample_weight=sample_weight,
                    message_clsname="Voting",
                    message=self._log_message(names[idx], idx + 1, len(clfs)),
                    cloning_function=clone_with_n_jobs_1,
                    force_refit=refit_estimators,
                )
                for idx, clf in enumerate(clfs)
                if clf != "drop"
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
        predictions = _collect_predictions(self, X, "predict")

        return np.asarray(predictions).T
