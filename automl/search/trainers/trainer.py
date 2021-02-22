from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from ..tuners.tuner import Tuner
from ..tuners.TPETuner import OptunaTPETuner
from ..blueprints.pipeline import create_pipeline_blueprint
from ..stage import AutoMLStage
from ..cv import get_cv_for_problem_type
from ...components.component import ComponentLevel, ComponentConfig
from ...problems.problem_type import ProblemType
from ...utils import validate_type

import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        problem_type: ProblemType,
        cv: Optional[Union[BaseCrossValidator, int]] = None,
        categorical_columns: Optional[list] = None,
        numeric_columns: Optional[list] = None,
        level: ComponentLevel = ComponentLevel.COMMON,
        tuner: Tuner = OptunaTPETuner,
        random_state=None,
    ) -> None:
        self.problem_type = problem_type
        self.cv = cv
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.level = level
        self.tuner = tuner
        self.random_state = random_state

    def _get_cv(self, problem_type: ProblemType, cv: Union[BaseCrossValidator, int]):
        validate_type(cv, "cv", (BaseCrossValidator, int, None))
        return get_cv_for_problem_type(problem_type, n_splits=cv)

    def fit(self, X, y, groups=None):
        self.pipeline_blueprint_ = create_pipeline_blueprint(
            problem_type=self.problem_type,
            categorical_columns=self.categorical_columns,
            numeric_columns=self.numeric_columns,
            level=self.level,
            X=X,
            y=y,
        )

        self.cv_ = self._get_cv(self.problem_type, self.cv)

        self.tuner_ = self.tuner(
            problem_type=self.problem_type,
            pipeline_blueprint=self.pipeline_blueprint_,
            random_state=self.random_state,
            cv=self.cv_,
        )

        self.tuner_.fit(X, y, groups=groups)

        return self
