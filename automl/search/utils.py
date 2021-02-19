from automl.components.estimators.linear_model.logistic_regression import (
    LogisticRegression,
)
from typing import Optional, Union

from ..components.estimators.tree.decision_tree import DecisionTreeClassifier
from .stage import AutoMLStage
from ..problems.problem_type import ProblemType
from ..components.transformers import *
from ..components.estimators import *
from ..components.flow import *
from ..components.component import ComponentLevel, ComponentConfig
from ..utils import validate_type

from sklearn.compose import make_column_selector


def create_pipeline_blueprint(
    problem_type: ProblemType,
    categorical_columns: Optional[list] = None,
    numeric_columns: Optional[list] = None,
    level: ComponentLevel = ComponentLevel.COMMON,
):
    if categorical_columns is None:
        categorical_columns = make_column_selector(dtype_include="category")

    if numeric_columns is None:
        numeric_columns = make_column_selector(dtype_include=[int, float])

    validate_type(problem_type, "problem_type", ProblemType)
    validate_type(level, "level", ComponentLevel)

    # steps in [] are tunable

    pipeline_steps_X = [
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
    pipeline_steps_y = [LabelEncoder()]

    pipeline_X = Pipeline(steps=pipeline_steps_X)
    pipeline_X.remove_invalid_components(
        pipeline_config=ComponentConfig(
            level=level,
            problem_type=problem_type,
        ),
        current_stage=AutoMLStage.PREPROCESSING,
    )

    return pipeline_X, pipeline_steps_y


def convert_tuning_grid(
    grid: Union[list, dict], param_dict: dict, level: str = ""
) -> None:
    if isinstance(grid, list):
        param_dict[level] = grid
        return
    for k, v in grid.items():
        convert_tuning_grid(v, param_dict, f"{level+'__' if level else ''}{k}")
    return