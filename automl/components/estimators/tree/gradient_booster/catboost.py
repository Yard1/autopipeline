from .gradient_booster_estimator import GradientBoosterEstimator
from .....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionDistribution,
)
from ....component import ComponentLevel
from .....problems import ProblemType

from automl_models.components.estimators.tree.gradient_booster.catboost import (
    CatBoostClassifierWithAutoCatFeatures,
    CatBoostRegressorWithAutoCatFeatures,
)

# tuning based on https://github.com/microsoft/FLAML/blob/main/flaml/model.py


def get_catboost_n_estimators(config, space):
    X = config.X
    if X is None:
        return IntUniformDistribution(10, 100, log=True)
    return IntUniformDistribution(10, min(2048, int(X.shape[0])), log=True)


class CatBoostClassifierBinary(GradientBoosterEstimator):
    _component_class = CatBoostClassifierWithAutoCatFeatures

    _default_parameters = {
        "n_estimators": 100,
        "learning_rate": None,
        "max_depth": 6,
        "task_type": "CPU",
        "verbose": False,
        "random_state": None,
        "auto_class_weights": "Balanced",
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(
            get_catboost_n_estimators, cost_bounds="upper"
        ),
        "max_depth": IntUniformDistribution(4, 10, cost_bounds="upper"),
        # "learning_rate": UniformDistribution(lower=0.005, upper=0.2, log=True),
    }
    _default_tuning_grid_extended = {
        "auto_class_weights": CategoricalDistribution(
            ["Balanced", None], cost_related=False
        )
    }

    _problem_types = {
        ProblemType.BINARY,
    }

    _component_level = ComponentLevel.COMMON
    _has_own_cat_encoding = True


class CatBoostClassifierMulticlass(GradientBoosterEstimator):
    _component_class = CatBoostClassifierWithAutoCatFeatures

    _default_parameters = {
        "n_estimators": 100,
        "learning_rate": None,
        "max_depth": 6,
        "task_type": "CPU",
        "verbose": False,
        "random_state": None,
        "auto_class_weights": "Balanced",
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(
            get_catboost_n_estimators, cost_bounds="upper"
        ),
        "max_depth": IntUniformDistribution(4, 10, cost_bounds="upper"),
        "learning_rate": UniformDistribution(
            lower=0.005, upper=0.2, log=True, cost_bounds="lower"
        ),
    }
    _default_tuning_grid_extended = {
        "auto_class_weights": CategoricalDistribution(
            ["Balanced", None], cost_related=False
        )
    }

    _problem_types = {
        ProblemType.MULTICLASS,
    }

    _component_level = ComponentLevel.COMMON
    _has_own_cat_encoding = True


class CatBoostRegressor(GradientBoosterEstimator):
    _component_class = CatBoostRegressorWithAutoCatFeatures

    _default_parameters = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "task_type": "CPU",
        "verbose": False,
        "random_state": None,
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(
            get_catboost_n_estimators, cost_bounds="upper"
        ),
        "max_depth": IntUniformDistribution(4, 10, cost_bounds="upper"),
        "learning_rate": UniformDistribution(
            lower=0.005, upper=0.2, log=True, cost_bounds="lower"
        ),
    }
    _default_tuning_grid_extended = {}

    _problem_types = {ProblemType.REGRESSION}

    _component_level = ComponentLevel.COMMON
    _has_own_cat_encoding = True
