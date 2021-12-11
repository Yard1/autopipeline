from category_encoders.cat_boost import CatBoostEncoder as _CatBoostEncoder
from category_encoders.wrapper import PolynomialWrapper

from .encoder import Encoder
from ..transformer import DataType
from ...component import ComponentLevel, ComponentConfig
from ....problems import ProblemType
from ....search.stage import AutoMLStage

from automl_models.components.transformers.encoder.kfold_encoder_wrapper import (
    KFoldEncoderWrapper,
)


class CatBoostEncoderBinary(Encoder):
    _component_class = KFoldEncoderWrapper
    _default_parameters = {
        "base_transformer": _CatBoostEncoder(
            **{
                "verbose": 0,
                "cols": [],
                "drop_invariant": False,
                "return_df": True,
                "handle_unknown": "value",
                "handle_missing": "value",
                "random_state": 0,
                "sigma": None,
                "a": 1,
            }
        ),
        "cv": 5,
        "return_same_type": True,
        "is_classification": True,
        "random_state": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY
    _problem_types = {
        ProblemType.BINARY,
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )


class CatBoostEncoderMulticlass(Encoder):
    _component_class = KFoldEncoderWrapper
    _default_parameters = {
        "base_transformer": PolynomialWrapper(
            _CatBoostEncoder(
                **{
                    "verbose": 0,
                    "cols": [],
                    "drop_invariant": False,
                    "return_df": True,
                    "handle_unknown": "value",
                    "handle_missing": "value",
                    "random_state": 0,
                    "sigma": None,
                    "a": 1,
                }
            )
        ),
        "cv": 5,
        "return_same_type": True,
        "is_classification": True,
        "random_state": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY
    _problem_types = {
        ProblemType.MULTICLASS,
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )


class CatBoostEncoderRegression(Encoder):
    _component_class = KFoldEncoderWrapper
    _default_parameters = {
        "base_transformer": _CatBoostEncoder(
            **{
                "verbose": 0,
                "cols": [],
                "drop_invariant": False,
                "return_df": True,
                "handle_unknown": "value",
                "handle_missing": "value",
                "random_state": 0,
                "sigma": None,
                "a": 1,
            }
        ),
        "cv": 5,
        "return_same_type": True,
        "is_classification": False,
        "random_state": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.NECESSARY
    _problem_types = {
        ProblemType.REGRESSION,
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or not getattr(config.estimator, "_has_own_cat_encoding", False)
        )
