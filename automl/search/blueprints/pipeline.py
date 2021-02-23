from typing import Optional

from automl import components

from ...components.estimators.tree.decision_tree import DecisionTreeClassifier
from ..stage import AutoMLStage
from ...problems.problem_type import ProblemType
from ...components.transformers import *
from ...components.estimators import *
from ...components.flow import *
from ...components.component import Component, ComponentLevel, ComponentConfig
from ...components.estimators.tree.tree_estimator import TreeEstimator
from ...utils import validate_type

from sklearn.compose import make_column_selector

from automl.components import estimators


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

    passthrough = {
        "Passthrough": Passthrough(),
        "Passthrough_Scaler": Passthrough(
            validity_condition=lambda config, stage: (
                config.estimator is None or isinstance(config.estimator, TreeEstimator)
            )
        ),
    }
    imbalance = {"AutoSMOTE": AutoSMOTE()}
    numeric_imputers = {
        "SimpleNumericImputer": SimpleNumericImputer(),
    }
    scalers_normalizers = {
        "StandardScaler": StandardScaler(),
    }
    categorical_imputers = {
        "SimpleCategoricalImputer": SimpleCategoricalImputer(),
    }
    categorical_encoders = {
        "OneHotEncoder": OneHotEncoder(),
    }
    estimators = {
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(),
    }
    components = {
        **passthrough,
        **imbalance,
        **numeric_imputers,
        **scalers_normalizers,
        **categorical_imputers,
        **categorical_encoders,
        **estimators,
    }

    pipeline_steps = [
        ("Imbalance", list(imbalance.values()) + [components["Passthrough"]]),
        (
            "ColumnTransformer",
            ColumnTransformer(
                transformers=[
                    (
                        "Categorical",
                        Pipeline(
                            steps=[
                                (
                                    "Imputer",
                                    list(numeric_imputers.values()),
                                ),
                                (
                                    "CategoricalEncoder",
                                    list(categorical_encoders.values()),
                                ),
                            ]
                        ),
                        categorical_columns,
                    ),
                    (
                        "Numeric",
                        Pipeline(
                            steps=[
                                (
                                    "Imputer",
                                    list(categorical_imputers.values()),
                                ),
                                (
                                    "ScalerNormalizer",
                                    list(scalers_normalizers.values())
                                    + [components["Passthrough_Scaler"]],
                                ),
                            ]
                        ),
                        numeric_columns,
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
            components["StandardScaler"],
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
            components["StandardScaler"],
            {},
        ),
        "Estimator": (components["LogisticRegression"], {"C": 1.0}),
    }

    pipeline = TopPipeline(
        steps=pipeline_steps,
        # preset_configurations=[d, d2]
    )
    pipeline.remove_invalid_components(
        pipeline_config=ComponentConfig(
            level=level,
            problem_type=problem_type,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            X=X,
            y=y,
        ),
        current_stage=AutoMLStage.PREPROCESSING,
    )

    return pipeline
