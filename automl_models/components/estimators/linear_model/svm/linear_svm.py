from sklearn.svm import LinearSVC as _LinearSVC, LinearSVR as _LinearSVR
from sklearn.utils.validation import check_is_fitted


# TODO: libsvm for small datasets
class LinearSVCCombinedPenaltyLossDynamicDual(_LinearSVC):
    def __init__(
        self,
        penalty_loss="l2-squared_hinge",
        *,
        tol=1e-4,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=200,
    ):
        self.dual = True
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.penalty_loss = penalty_loss

    def fit(self, X, y, sample_weight=None):
        if self.penalty_loss == "l2-hinge":
            self.dual = True
        else:
            self.dual = False
        super().fit(X, y, sample_weight=sample_weight)

    @property
    def penalty_loss(self):
        return self._penalty_loss

    @penalty_loss.setter
    def penalty_loss(self, value):
        self._penalty_loss = value
        self.penalty, self.loss = value.split("-")

    def get_params(self, deep=True):
        r = super().get_params(deep=deep)
        r["penalty_loss"] = self.penalty_loss
        return r

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        return super()._predict_proba_lr(X)


class LinearSVRDynamicDual(_LinearSVR):
    def __init__(
        self,
        *,
        epsilon=0.0,
        tol=1e-4,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        verbose=0,
        random_state=None,
        max_iter=200,
    ):
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.dual = True
        self.loss = loss

    def fit(self, X, y, sample_weight=None):
        if self.loss == "epsilon_insensitive":
            self.dual = True
        else:
            self.dual = False
        super().fit(X, y, sample_weight=sample_weight)
