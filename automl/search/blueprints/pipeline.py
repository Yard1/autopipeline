import numpy as np

from automl.components.transformers.feature_selector.boruta import (
    BorutaSHAPClassification,
    BorutaSHAPRegression,
)
from typing import Optional

from automl import components

from ...components.estimators.tree.decision_tree import DecisionTreeClassifier
from ..stage import AutoMLStage
from ...problems.problem_type import ProblemType
from ...components.flow import ColumnTransformer, Pipeline, TopPipeline
from ...components.transformers import *
from ...components.estimators import *
from ...components.component import Component, ComponentLevel, ComponentConfig
from ...components.estimators.tree.tree_estimator import TreeEstimator
from ...utils import validate_type

from .column_selector import make_column_selector


def _scaler_passthrough_condition(config, stage):
    return config.estimator is None or isinstance(config.estimator, TreeEstimator)


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
    numeric_imputers = {
        "SimpleNumericImputer": SimpleNumericImputer(),
    }
    scalers_normalizers = {
        "CombinedScalerTransformer": CombinedScalerTransformer(),
    }
    categorical_imputers = {
        "SimpleCategoricalImputer": SimpleCategoricalImputer(),
    }
    categorical_encoders = {
        "OneHotEncoder": OneHotEncoder(),
    }
    oridinal_encoder = {"OrdinalEncoder": OrdinalEncoder()}
    feature_selectors = {
        "BorutaSHAPClassification": BorutaSHAPClassification(),
        "BorutaSHAPRegression": BorutaSHAPRegression(),
    }
    estimators = {
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "LogisticRegression": LogisticRegression(),
        "LGBMClassifier": LGBMClassifier(),
        "LGBMRegressor": LGBMRegressor(),
        "RandomForestClassifier": RandomForestClassifier(),
        "RandomForestRegressor": RandomForestRegressor(),
    }
    components = {
        **passthrough,
        **imbalance,
        **numeric_imputers,
        **scalers_normalizers,
        **categorical_imputers,
        **categorical_encoders,
        **oridinal_encoder,
        **feature_selectors,
        **estimators,
    }

    pipeline_steps = [
        ("Imbalance", list(imbalance.values()) + [components["Passthrough"]]),
        (
            "ColumnImputation",
            ColumnTransformer(
                transformers=[
                    (
                        "CategoricalImputer",
                        list(categorical_imputers.values()),
                        categorical_selector,
                    ),
                    (
                        "NumericImputer",
                        list(numeric_imputers.values()),
                        numeric_selector,
                    ),
                ],
            ),
        ),
        (
            "FeatureSelector",
            list(feature_selectors.values()) + [components["Passthrough"]],
        ),
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
            "Estimator",
            list(estimators.values()),
        ),
    ]

    d = {
        "Preprocessor__ColumnTransformer__Categorical__Imputer": (
            components["SimpleCategoricalImputer"],
            {},
        ),
        "Preprocessor__ColumnTransformer__Categorical__CategoricalEncoder": (
            components["OneHotEncoder"],
            {},
        ),
        "Preprocessor__ColumnTransformer__Numeric__Imputers": (
            components["SimpleNumericImputer"],
            {},
        ),
        "Preprocessor__ColumnTransformer__Numeric__ScalerNormalizer": (
            components["CombinedScalerTransformer"],
            {},
        ),
        "Estimator": (components["LogisticRegression"], {"C": 4.0}),
    }
    d2 = {
        "Preprocessor__ColumnTransformer__Categorical__CategoricalEncoder": (
            components["OneHotEncoder"],
            {},
        ),
        "Preprocessor__ColumnTransformer__Numeric__ScalerNormalizer": (
            components["CombinedScalerTransformer"],
            {},
        ),
        "Estimator": (components["LogisticRegression"], {"C": 1.0}),
    }

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
    pipeline.call_tuning_grid_funcs(config=config, stage=AutoMLStage.PREPROCESSING)
    return pipeline
