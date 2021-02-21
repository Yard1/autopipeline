from typing import Optional

from automl import components

from ...components.estimators.tree.decision_tree import DecisionTreeClassifier
from ..stage import AutoMLStage
from ...problems.problem_type import ProblemType
from ...components.transformers import *
from ...components.estimators import *
from ...components.flow import *
from ...components.component import Component, ComponentLevel, ComponentConfig
from ...utils import validate_type

from sklearn.compose import make_column_selector

from automl.components import estimators


def create_pipeline_blueprint(
    problem_type: ProblemType,
    categorical_columns: Optional[list] = None,
    numeric_columns: Optional[list] = None,
    missing_values: bool = True,
    level: ComponentLevel = ComponentLevel.COMMON,
) -> TopPipeline:
    if categorical_columns is None:
        categorical_columns = make_column_selector(dtype_include="category")

    if numeric_columns is None:
        numeric_columns = make_column_selector(dtype_include=[int, float])

    validate_type(problem_type, "problem_type", ProblemType)
    validate_type(level, "level", ComponentLevel)

    # steps in [] are tunable

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
        **numeric_imputers,
        **scalers_normalizers,
        **categorical_imputers,
        **categorical_encoders,
        **estimators,
    }

    pipeline_steps = [
        (
            "Preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "Categorical",
                        Pipeline(
                            steps=[
                                ("Imputer", list(numeric_imputers.values())),
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
                                ("Imputer", list(categorical_imputers.values())),
                                (
                                    "ScalerNormalizer",
                                    list(scalers_normalizers.values()),
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
        "Preprocessor__Categorical__Imputer": (
            components["SimpleCategoricalImputer"],
            {},
        ),
        "Preprocessor__Categorical__CategoricalEncoder": (
            components["OneHotEncoder"],
            {},
        ),
        "Preprocessor__Numeric__Imputers": (
            components["SimpleNumericImputer"],
            {},
        ),
        "Preprocessor__Numeric__ScalerNormalizer": (components["StandardScaler"], {}),
        "Estimator": (components["LogisticRegression"], {"C": 4.0}),
    }
    d2 = {
        "Preprocessor__Categorical__CategoricalEncoder": (
            components["OneHotEncoder"],
            {},
        ),
        "Preprocessor__Numeric__ScalerNormalizer": (components["StandardScaler"], {}),
        "Estimator": (components["LogisticRegression"], {"C": 1.0}),
    }

    pipeline = TopPipeline(steps=pipeline_steps, preset_configurations=[d, d2])
    pipeline.remove_invalid_components(
        pipeline_config=ComponentConfig(
            level=level,
            problem_type=problem_type,
            missing_values=missing_values,
        ),
        current_stage=AutoMLStage.PREPROCESSING,
    )

    return pipeline
