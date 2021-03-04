from automl.search.distributions.distributions import CategoricalDistribution
from sklearn.svm import LinearSVC as _LinearSVC, LinearSVR as _LinearSVR
from .svm import SVM
from .....problems import ProblemType
from .....search.distributions import UniformDistribution


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
        max_iter=1000,
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


class LinearSVC(SVM):
    _component_class = LinearSVCCombinedPenaltyLossDynamicDual

    _default_parameters = {
        "penalty_loss": "l2-squared_hinge",
        "tol": 1e-4,
        "C": 1.0,
        "multi_class": "ovr",
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "verbose": 0,
        "random_state": None,
        "max_iter": 20000,
    }

    _default_tuning_grid = {
        "penalty_loss": CategoricalDistribution(
            ["l2-squared_hinge", "l1-squared_hinge", "l2-hinge"]
        ),
        "C": UniformDistribution(0.01, 10),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}


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
        max_iter=1000,
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
        if self.penalty_loss == "epsilon_insensitive":
            self.dual = True
        else:
            self.dual = False
        super().fit(X, y, sample_weight=sample_weight)


class LinearSVR(SVM):
    _component_class = LinearSVRDynamicDual

    _default_parameters = {
        "epsilon": 0.001,
        "tol": 1e-4,
        "C": 1.0,
        "loss": "epsilon_insensitive",
        "fit_intercept": True,
        "intercept_scaling": 1,
        "verbose": 0,
        "random_state": None,
        "max_iter": 20000,
    }

    _default_tuning_grid = {
        "loss": CategoricalDistribution(
            ["epsilon_insensitive", "squared_epsilon_insensitive"]
        ),
        "C": UniformDistribution(0.01, 10),  # consider log like in autosklearn?
        "epsilon": UniformDistribution(0.001, 1, log=True),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.REGRESSION}
