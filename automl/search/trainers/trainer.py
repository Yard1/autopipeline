from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import gc

from collections import defaultdict

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from ..tuners.utils import ray_context
from ..tuners.tuner import Tuner
from ..tuners.OptunaTPETuner import OptunaTPETuner
from ..tuners.blendsearch import BlendSearchTuner
from ..tuners.BOHBTuner import BOHBTuner
from ..tuners.HEBOTuner import HEBOTuner
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
        tuner: Tuner = BlendSearchTuner,
        best_percentile: int = 90,
        stacking_level: int = 0,
        early_stopping: bool = False,
        cache: Union[str, bool] = False,
        random_state=None,
        tune_kwargs: dict = None,
    ) -> None:
        self.problem_type = problem_type
        self.cv = cv
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.level = level
        self.tuner = tuner
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.cache = cache
        self.best_percentile = best_percentile
        self.stacking_level = stacking_level
        self.tune_kwargs = tune_kwargs or {}

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
            early_stopping=self.early_stopping,
            cache=self.cache,
        )

        gc.collect()
        print("starting tuning")
        with ray_context(
            global_checkpoint_s=self.tune_kwargs.pop("TUNE_GLOBAL_CHECKPOINT_S", 10)
        ):
            self.tuner_.fit(X, y, groups=groups)
            return self
            self.secondary_tuner = HEBOTuner
            self.secondary_tuners_ = []

            results = self.tuner_.analysis_.results
            results_df = self.tuner_.analysis_.results_df
            if "dataset_fraction_" in results_df.columns:
                results_df = results_df[results_df["dataset_fraction_"] >= 1.0]
            percentile = np.percentile(
                results_df["mean_test_score"], self.best_percentile
            )
            grouped_results_df = results_df.groupby(by="config.Estimator")
            for name, group in grouped_results_df:
                if group["mean_test_score"].max() >= percentile:
                    group_trial_ids = set(group.index)
                    known_points = [
                        (
                            {
                                config_key: config_value
                                for config_key, config_value in v["config"].items()
                                if config_value != "passthrough"
                            },
                            v["mean_test_score"],
                        )
                        for k, v in results.items()
                        if k in group_trial_ids
                    ]
                    self.secondary_tuners_.append(
                        self.secondary_tuner(
                            problem_type=self.problem_type,
                            pipeline_blueprint=self.pipeline_blueprint_,
                            random_state=self.random_state,
                            cv=self.cv_,
                            early_stopping=False,
                            cache=self.cache,
                            known_points=known_points,
                        )
                    )
                    self.secondary_tuners_[-1].fit(X, y, groups=groups)
        return self
