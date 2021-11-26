import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel

EPS = 1e-10


def _accumulate_prediction(e, X):
    return e.predict(X, check_input=False)


class RandomForestRegressorWithStd(RandomForestRegressor):
    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        predictions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_accumulate_prediction)(e, X) for e in self.estimators_
        )

        mean = np.mean(predictions, axis=0)
        var = np.var(predictions, axis=0)

        var[var < EPS] = EPS
        var[np.isnan(var)] = EPS

        return mean, var
