import numpy as np

from lightgbm.sklearn import (
    LGBMClassifier as _LGBMClassifier,
    LGBMRegressor as _LGBMRegressor,
)
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

# tuning based on https://github.com/microsoft/FLAML/blob/main/flaml/model.py


def get_lgbm_n_estimators(config, space):
    X = config.X
    if X is None:
        return IntUniformDistribution(10, 200, log=True)
    return IntUniformDistribution(10, min(8192, int(X.shape[0])), log=True)


class LGBMClassifier(GradientBoosterEstimator):
    _component_class = _LGBMClassifier

    _default_parameters = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample_for_bin": 200000,
        "class_weight": "balanced",
        "min_split_gain": 0,
        "min_child_weight": 1e-3,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 1 / 1024,
        "reg_lambda": 1.0,
        "random_state": 0,
        "n_jobs": 1,
        "verbose": -1,
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(get_lgbm_n_estimators, cost_bounds="upper"),
        "num_leaves": IntUniformDistribution(2, 256, cost_bounds="upper"),
        "learning_rate": UniformDistribution(1 / 1024, 0.3, log=True, cost_bounds="lower"),
    }
    _default_tuning_grid_extended = {
        "subsample": UniformDistribution(0.1, 1.0, cost_bounds="upper"),
        "colsample_bytree": UniformDistribution(0.01, 1.0, cost_bounds="upper"),
        "reg_alpha": UniformDistribution(1 / 1024, 1024, log=True, cost_related=False),
        "reg_lambda": UniformDistribution(1 / 1024, 1024, log=True, cost_related=False),
        "min_child_samples": IntUniformDistribution(
            2, 2 ** 7, log=True, cost_related=False
        ),
        "class_weight": CategoricalDistribution([None, "balanced"], cost_related=False),
    }

    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }

    _component_level = ComponentLevel.COMMON
    _has_own_cat_encoding = True


class LGBMRegressor(GradientBoosterEstimator):
    _component_class = _LGBMRegressor

    _default_parameters = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample_for_bin": 200000,
        "min_split_gain": 0,
        "min_child_weight": 1e-3,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 1 / 1024,
        "reg_lambda": 1.0,
        "random_state": 0,
        "n_jobs": 1,
        "verbose": -1,
    }

    _default_tuning_grid = {
        "n_estimators": FunctionDistribution(get_lgbm_n_estimators, cost_bounds="upper"),
        "num_leaves": IntUniformDistribution(2, 256, cost_bounds="upper"),
        "learning_rate": UniformDistribution(1 / 1024, 0.3, log=True, cost_bounds="lower"),
    }
    _default_tuning_grid_extended = {
        "subsample": UniformDistribution(0.1, 1.0, cost_bounds="upper"),
        "colsample_bytree": UniformDistribution(0.01, 1.0, cost_bounds="upper"),
        "reg_alpha": UniformDistribution(1 / 1024, 1024, log=True, cost_related=False),
        "reg_lambda": UniformDistribution(1 / 1024, 1024, log=True, cost_related=False),
        "min_child_samples": IntUniformDistribution(
            2, 2 ** 7, log=True, cost_related=False
        ),
    }

    _problem_types = {
        ProblemType.REGRESSION,
    }

    _component_level = ComponentLevel.COMMON
    _has_own_cat_encoding = True


# TODO: add dart, goss
# TODO add expert configs from LightAutoML https://github.com/sberbank-ai-lab/LightAutoML/blob/master/lightautoml/ml_algo/boost_lgbm.py
