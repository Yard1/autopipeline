from typing import Optional, Union

from ..components.estimators.tree.decision_tree import DecisionTreeClassifier
from ..components.estimators.linear_model.logistic_regression import LogisticRegression
from .stage import AutoMLStage
from ..problems.problem_type import ProblemType
from ..components.transformers import *
from ..components.estimators import *
from ..components.flow import *
from ..components.component import Component, ComponentLevel, ComponentConfig
from ..utils import validate_type

from sklearn.compose import make_column_selector


def call_component_if_needed(possible_component):
    if isinstance(possible_component, Component):
        return possible_component()
    else:
        return possible_component

def create_pipeline_blueprint(
    problem_type: ProblemType,
    categorical_columns: Optional[list] = None,
    numeric_columns: Optional[list] = None,
    level: ComponentLevel = ComponentLevel.COMMON,
) -> TopPipeline:
    if categorical_columns is None:
        categorical_columns = make_column_selector(dtype_include="category")

    if numeric_columns is None:
        numeric_columns = make_column_selector(dtype_include=[int, float])

    validate_type(problem_type, "problem_type", ProblemType)
    validate_type(level, "level", ComponentLevel)

    # steps in [] are tunable

    pipeline_steps = [
        (
            "Preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "Categorical",
                        Pipeline(
                            steps=[
                                ("Imputer", [SimpleCategoricalImputer()]),
                                ("CategoricalEncoder", [OneHotEncoder()]),
                            ]
                        ),
                        categorical_columns,
                    ),
                    (
                        "Numeric",
                        Pipeline(
                            steps=[
                                ("Imputer", [SimpleNumericImputer()]),
                                ("ScalerNormalizer", [StandardScaler(), Passthrough()]),
                            ]
                        ),
                        numeric_columns,
                    ),
                ],
            ),
        ),
        ("Estimator", [DecisionTreeClassifier(), LogisticRegression()]),
    ]

    pipeline = TopPipeline(steps=pipeline_steps)
    pipeline.remove_invalid_components(
        pipeline_config=ComponentConfig(
            level=level,
            problem_type=problem_type,
        ),
        current_stage=AutoMLStage.PREPROCESSING,
    )

    return pipeline