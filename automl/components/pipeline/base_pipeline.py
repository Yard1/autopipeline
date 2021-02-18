from typing import Iterable
from copy import deepcopy
from sklearn.pipeline import Pipeline, make_pipeline

from ..transformers import *
from ..estimators import *
from ...problems import ProblemType
from ..uitls import get_step_choice_grid


def make_pipeline_if_necessary(steps):
    if len(steps) == 1:
        return steps[0]()
    return Pipeline(steps=[(str(step), step()) for step in steps])

class Pipeline(Transformer):
    _component_class = Pipeline
    def __call__(self):
        params = deepcopy(self.final_parameters)
        steps = [(name, step()) for name, step in params["steps"]]
        params["steps"] = steps

        return self._component_class(**params)

    def get_tuning_grid(self, use_extended: bool = False) -> dict:
        default_grid = super().get_tuning_grid(use_extended=use_extended)
        step_grids = {name:get_step_choice_grid(step) for name, step in self.final_parameters["steps"]}
        return {**step_grids, **default_grid}
class BasePipeline(Pipeline):
    @staticmethod
    def get_pipeline_blueprint(
        categorical_columns: list,
        numerical_columns: list,
        problem_type: ProblemType,
        estimator: Estimator,
    ):
        pipeline_steps_X = [
            ("Preprocess", ColumnTransformer(
                transformers=[
                    (
                        "Categorical",
                        Pipeline(
                            steps=[
                                ("Imputer", SimpleCategoricalImputer()),
                                ("CategoricalEncoder", OneHotEncoder()),
                            ]
                        ),
                        categorical_columns,
                    ),
                    (
                        "Numerical",
                        Pipeline(
                            steps=[
                                ("Imputer", SimpleNumericalImputer()),
                            ]
                        ),
                        numerical_columns,
                    ),
                ],
            )),
        ]
        pipeline_steps_y = [LabelEncoder()]

        return Pipeline(steps=pipeline_steps_X), pipeline_steps_y

    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self._validate_steps()
