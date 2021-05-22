from .feature_selector import FeatureSelector
from ...component import ComponentLevel, ComponentConfig
from ...estimators.tree.tree_estimator import TreeEstimator
from ....problems import ProblemType
from ....search.stage import AutoMLStage

from automl_models.components.transformers.feature_selector.select_from_model import (
    PandasSHAPSelectFromModel,
)


class SHAPSelectFromModelClassification(FeatureSelector):
    _component_class = PandasSHAPSelectFromModel
    _default_parameters = {
        "estimator": "LGBMClassifier",
        "threshold": "mean",
        "prefit": False,
        "norm_order": 1,
        "max_features": None,
        "importance_getter": "auto",
        "n_estimators": "auto",
        "random_state": 0,
    }
    _component_level = ComponentLevel.COMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        # RARE turns on Boruta which does the same thing but better (at a run time cost)
        return (
            super_check
            and (config.level is None or (config.level < ComponentLevel.RARE))
            and (
                config.estimator is None
                or not isinstance(config.estimator, TreeEstimator)
            )
        )


class SHAPSelectFromModelRegression(FeatureSelector):
    _component_class = PandasSHAPSelectFromModel
    _default_parameters = {
        "estimator": "LGBMRegressor",
        "threshold": "mean",
        "prefit": False,
        "norm_order": 1,
        "max_features": None,
        "importance_getter": "auto",
        "n_estimators": "auto",
        "random_state": 0,
    }
    _component_level = ComponentLevel.COMMON
    _problem_types = {ProblemType.REGRESSION}

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        # RARE turns on Boruta which does the same thing but better (at a run time cost)
        return (
            super_check
            and (config.level is None or (config.level < ComponentLevel.RARE))
            and (
                config.estimator is None
                or not isinstance(config.estimator, TreeEstimator)
            )
        )
