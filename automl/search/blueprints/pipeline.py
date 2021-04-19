from automl.components.transformers.knn.knn_transformer import NCATransformer
import numpy as np

from typing import Optional

from ..stage import AutoMLStage
from ...problems.problem_type import ProblemType
from ...components.flow import (
    ColumnTransformer,
    make_column_selector,
    Pipeline,
    TopPipeline,
)
from ...components.transformers import *
from ...components.estimators import *
from ...components.component import Component, ComponentLevel, ComponentConfig
from ...components.estimators.linear_model.linear_model_estimator import (
    LinearModelEstimator,
)
from ...components.estimators.knn.knn_estimator import KNNEstimator
from ...utils import validate_type


def _scaler_passthrough_condition(config, stage):
    return config.estimator is None or not isinstance(
        config.estimator, (LinearModelEstimator, KNNEstimator)
    )


categorical_selector = make_column_selector(dtype_include="category")
numeric_selector = make_column_selector(dtype_exclude="category")


def create_pipeline_blueprint(
    problem_type: ProblemType,
    X=None,
    y=None,
    categorical_columns: Optional[list] = None,
    numeric_columns: Optional[list] = None,
    level: ComponentLevel = ComponentLevel.COMMON,
) -> TopPipeline:
    validate_type(problem_type, "problem_type", ProblemType)
    validate_type(level, "level", ComponentLevel)

    # steps in [] are tunable
    # there are three possible states for a step depending on an estimator:
    # - none of the components are valid
    # - one component is valid
    # - all components are valid
    # make sure that is the case!

    passthrough = {
        "Passthrough": Passthrough(),
        "Passthrough_Scaler": Passthrough(
            validity_condition=_scaler_passthrough_condition
        ),
    }
    imbalance = {"AutoSMOTE": AutoSMOTE()}
    imputers = {
        "CombinedSimpleImputer": CombinedSimpleImputer(),
        "IterativeImputer": IterativeImputer(),
    }
    scalers_normalizers = {
        "CombinedScalerTransformer": CombinedScalerTransformer(),
        "MinMaxScaler": MinMaxScaler(),
    }
    categorical_encoders = {
        #"OneHotEncoder": OneHotEncoder(),
        "CatBoostEncoderBinary": CatBoostEncoderBinary(),
        "CatBoostEncoderMulticlass": CatBoostEncoderMulticlass(),
        "CatBoostEncoderRegression": CatBoostEncoderRegression(),
        "BayesianTargetEncoderBinary": BayesianTargetEncoderBinary(),
        "BayesianTargetEncoderMulticlass": BayesianTargetEncoderMulticlass(),
        "BayesianTargetEncoderRegression": BayesianTargetEncoderRegression(),
    }
    oridinal_encoder = {"OrdinalEncoder": OrdinalEncoder()}
    feature_selectors = {
        "BorutaSHAPClassification": BorutaSHAPClassification(),
        "BorutaSHAPRegression": BorutaSHAPRegression(),
        "SHAPSelectFromModelClassification": SHAPSelectFromModelClassification(),
        "SHAPSelectFromModelRegression": SHAPSelectFromModelRegression(),
    }
    svm_kernels = {
        "NystroemRBF": NystroemRBF(),
        "PolynomialCountSketch": PolynomialCountSketch(),
        "NystroemSigmoid": NystroemSigmoid(),
    }
    knn_transformers = {
        "KNNTransformer": KNNTransformer(),
        # "NCATransformer": NCATransformer(),
    }
    estimators = {
        # "DecisionTreeClassifier": DecisionTreeClassifier(),
        # "DecisionTreeRegressor": DecisionTreeRegressor(),
        "LogisticRegression": LogisticRegression(),
        "LogisticRegression_L1": LogisticRegression(l1_ratio=1),
        "LogisticRegression_EN": LogisticRegression(l1_ratio=0.5),
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(),
        "LGBMClassifier": LGBMClassifier(),
        "LGBMRegressor": LGBMRegressor(),
        "RandomForestClassifier": RandomForestClassifier(),
        "RandomForestRegressor": RandomForestRegressor(),
        "ExtraTreesClassifier": RandomForestClassifier(randomization_type="et"),
        "ExtraTreesRegressor": RandomForestRegressor(randomization_type="et"),
        "LinearSVC": LinearSVC(),
        "LinearSVR": LinearSVR(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "KNeighborsRegressor": KNeighborsRegressor(),
    }
    components = {
        **passthrough,
        **imbalance,
        **imputers,
        **scalers_normalizers,
        **categorical_encoders,
        **oridinal_encoder,
        **feature_selectors,
        **svm_kernels,
        **knn_transformers,
        **estimators,
    }

    pipeline_steps = [
        (
            "FeatureSelector",
            [components["Passthrough"]] + list(feature_selectors.values()),
        ),
        ("Imputer", list(imputers.values())),
        ("Imbalance", [components["Passthrough"]] + list(imbalance.values())),
        (
            "ColumnOrdinal",
            ColumnTransformer(
                transformers=[
                    (
                        "OrdinalEncoder",
                        [components["OrdinalEncoder"]],
                        categorical_selector,
                    ),
                ],
            ),
        ),
        (
            "ColumnEncodingScaling",
            ColumnTransformer(
                transformers=[
                    (
                        "CategoricalEncoder",
                        list(categorical_encoders.values()),
                        categorical_selector,
                    ),
                    (
                        "ScalerNormalizer",
                        list(scalers_normalizers.values()),
                        numeric_selector,
                    ),
                ],
            ),
        ),
        (
            "SVMKernelApproximation",
            [components["Passthrough"]] + list(svm_kernels.values()),
        ),
        (
            "KNNTransformer",
            list(knn_transformers.values()),
        ),
        (
            "Estimator",
            list(estimators.values()),
        ),
    ]

    pipeline = TopPipeline(
        steps=pipeline_steps,
        # preset_configurations=[d, d2]
    )
    config = ComponentConfig(
        level=level,
        problem_type=problem_type,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        X=X,
        y=y,
    )
    pipeline.remove_invalid_components(
        pipeline_config=config,
        current_stage=AutoMLStage.PREPROCESSING,
    )
    # TODO: move to trainer
    pipeline.call_tuning_grid_funcs(config=config, stage=AutoMLStage.PREPROCESSING)
    return pipeline
