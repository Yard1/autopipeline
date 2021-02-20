from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from collections import defaultdict

from ..tuners.tuner import Tuner
from ..tuners.TPETuner import TPETuner
from ..utils import create_pipeline_blueprint
from ..stage import AutoMLStage
from ...components.component import ComponentLevel, ComponentConfig
from ...problems.problem_type import ProblemType

import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        problem_type: ProblemType,
        categorical_columns: Optional[list] = None,
        numeric_columns: Optional[list] = None,
        level: ComponentLevel = ComponentLevel.COMMON,
        tuner: Tuner = TPETuner,
        random_state=None,
    ) -> None:
        self.problem_type = problem_type
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.level = level
        self.tuner = tuner
        self.random_state = random_state

    def fit(self, X, y):
        self.pipeline_blueprint_ = create_pipeline_blueprint(
            self.problem_type,
            self.categorical_columns,
            self.numeric_columns,
            self.level,
        )

        self.tuner_ = self.tuner(
            pipeline_blueprint=self.pipeline_blueprint_,
            random_state=self.random_state,
        )

        self.tuner_.fit(X, y)
