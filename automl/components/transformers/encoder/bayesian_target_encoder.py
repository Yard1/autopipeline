from .encoder import Encoder
from ..transformer import DataType
from ...component import ComponentLevel, ComponentConfig
from ....search.stage import AutoMLStage
from ....problems import ProblemType

from automl_models.components.transformers.encoder.bayesian_target_encoder import (
    SamplingBayesianEncoder,
    TaskType,
)


class BayesianTargetEncoderBinary(Encoder):
    _component_class = SamplingBayesianEncoder
    _default_parameters = {
        "verbose": 0,
        "cols": None,
        "drop_invariant": True,
        "return_df": True,
        "handle_unknown": "value",
        "handle_missing": "value",
        "random_state": 0,
        "prior_samples_ratio": 1e-4,
        "n_draws": 5,
        "mapper": "identity",
        "task": TaskType.BINARY_CLASSIFICATION,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

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


class BayesianTargetEncoderMulticlass(Encoder):
    _component_class = SamplingBayesianEncoder
    _default_parameters = {
        "verbose": 0,
        "cols": None,
        "drop_invariant": False,
        "return_df": True,
        "handle_unknown": "value",
        "handle_missing": "value",
        "random_state": 0,
        "prior_samples_ratio": 1e-4,
        "n_draws": 5,
        "mapper": "identity",
        "task": TaskType.MULTICLASS_CLASSIFICATION,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

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


class BayesianTargetEncoderRegression(Encoder):
    _component_class = SamplingBayesianEncoder
    _default_parameters = {
        "verbose": 0,
        "cols": None,
        "drop_invariant": False,
        "return_df": True,
        "handle_unknown": "value",
        "handle_missing": "value",
        "random_state": 0,
        "prior_samples_ratio": 1e-4,
        "n_draws": 5,
        "mapper": "identity",
        "task": TaskType.REGRESSION,
        "n_jobs": None,
    }
    _allowed_dtypes = {DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

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
