from copy import deepcopy
from sklearn.utils.validation import check_is_fitted
from automl.utils.display import IPythonDisplay
from automl.utils.logging import make_header
from typing import Any, Dict, Optional, Union
from IPython.display import HTML

import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, BaseCrossValidator
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle, estimator_html_repr

import joblib
from ray.util.joblib import register_ray

from .trainers.trainer import Trainer
from .cv import get_cv_for_problem_type
from ..components import DataType, ComponentLevel
from ..problems import ProblemType
from ..components import LabelEncoder
from ..utils import validate_type
from .utils import flatten_iterable, get_obj_name, ray_context

from automl_models.components.preprocessing.prepare_data import (
    PrepareDataFrame,
    clean_df,
)

import ray
import warnings
import logging

logger = logging.getLogger(__name__)
register_ray()


# TODO: unique ID
class AutoML(BaseEstimator):
    def __init__(
        self,
        problem_type: Optional[Union[str, ProblemType]] = None,
        test_size: float = 0.25,
        cv: Union[int, BaseCrossValidator] = 5,
        stacking_cv: Union[int, BaseCrossValidator] = 5,
        level: Union[str, int, ComponentLevel] = ComponentLevel.COMMON,
        random_state: Optional[int] = None,  # TODO: support other random states
        target_metric=None,
        float_dtype: type = np.float32,
        int_dtype: Optional[type] = None,
        trainer_config: Optional[dict] = None,
    ) -> None:
        self.problem_type = problem_type
        self.test_size = test_size
        self.level = level
        self.cv = cv
        self.stacking_cv = stacking_cv
        self.random_state = random_state
        self.float_dtype = float_dtype
        self.target_metric = target_metric
        self.int_dtype = int_dtype
        self.trainer_config = trainer_config or {
            "cache": True,
            "early_stopping": True,
            "return_test_scores_during_tuning": True,
        }
        super().__init__()

        self._displays = {
            "results_display": IPythonDisplay("results_display"),
            "pipeline_display": IPythonDisplay("pipeline_display"),
        }

        self._validate()

    def _validate(self):
        validate_type(self.level, "level", (str, int, ComponentLevel))
        validate_type(self.problem_type, "problem_type", (str, ProblemType, type(None)))
        validate_type(
            self.cv, "cv", (int, BaseCrossValidator, _RepeatedSplits)
        )  # use check_cv instead
        validate_type(
            self.stacking_cv, "stacking_cv", (int, BaseCrossValidator, _RepeatedSplits)
        )  # use check_cv instead
        validate_type(self.test_size, "test_size", float)
        if isinstance(self.cv, int) and self.cv < 2:
            raise ValueError(
                f"If cv is an int, it must be bigger than 2. Got {self.cv}."
            )
        if isinstance(self.stacking_cv, int) and self.stacking_cv < 2:
            raise ValueError(
                f"If stacking_cv is an int, it must be bigger than 2. Got {self.stacking_cv}."
            )
        if self.test_size < 0 or self.test_size > 0.9:
            raise ValueError("test_size must be in range (0.0, 0.9)")
        if not is_float_dtype(self.float_dtype):
            raise TypeError(
                f"Expected float_dtype to be a float dtype, got {type(self.float_dtype)}"
            )
        if self.int_dtype is not None and not is_integer_dtype(self.int_dtype):
            raise TypeError(
                f"Expected int_dtype to be an integer dtype or None, got {type(self.int_dtype)}"
            )

    def _drop_y_nan_rows(self, X, y):
        y_reset_index = y.reset_index(drop=True)
        nan_ilocs = y_reset_index[pd.isnull(y_reset_index)].index
        if len(nan_ilocs) > 0:
            logger.warn(
                f"Dropping {len(nan_ilocs)} y rows with NaNs: {list(nan_ilocs)}"
            )
            X.drop(X.index[nan_ilocs], inplace=True)

    def _translate_problem_type(self, problem_type) -> ProblemType:
        if problem_type is None:
            return problem_type

        if isinstance(problem_type, ProblemType):
            return problem_type

        if problem_type == "classification":
            return "classification"
        try:
            return ProblemType.translate(problem_type)
        except ValueError:
            raise ValueError(
                f"Wrong problem_type! Expected one of {['regression', 'classification', 'binary', 'multiclass', None, ProblemType]} got {problem_type}"
            )

    def _determine_problem_type(
        self, problem_type: Optional[Union[str, ProblemType]], y: pd.Series
    ):
        determined_problem_type = None

        if DataType.is_categorical(y.dtype):
            determined_problem_type = (
                ProblemType.BINARY
                if len(y.cat.categories) == 2
                else ProblemType.MULTICLASS
            )
        elif DataType.is_numeric(y.dtype):
            determined_problem_type = ProblemType.REGRESSION
        elif problem_type is None:
            raise ValueError("Could not determine problem type.")
        else:
            logger.warn("Could not determine problem type.")

        logger.info(f"Problem type determined as {determined_problem_type}")

        if (
            determined_problem_type is not None
            and problem_type is not None
            and determined_problem_type != problem_type
        ):
            logger.warn(
                f"Determined problem type {determined_problem_type} doesn't match given problem type {problem_type}, forcing {problem_type}."
            )

            if problem_type == ProblemType.REGRESSION:
                y = y.astype(self.float_dtype)
            else:
                y = y.astype(pd.CategoricalDtype(y.unique()))
                if len(y.cat.categories) < 2:
                    raise ValueError(f"y has too few labels: {len(y.cat.categories)}")
                if (
                    len(y.cat.categories) == 2
                    and problem_type != ProblemType.MULTICLASS
                ):
                    raise ValueError(
                        f"y has too many labels for binary classification: {len(y.cat.categories)}"
                    )
            determined_problem_type = problem_type

        return determined_problem_type, y

    def _make_test_split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_seed_,
            stratify=y if self.problem_type_.is_classification() else None,
        )

    def _shuffle_data(self, X, y):
        X, y = shuffle(
            X,
            y,
            random_state=self.random_seed_,
        )
        return X, y

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        ordinal_columns: Optional[Dict[str, list]] = None,
    ):
        logger.info(make_header("AutoML Fit"))

        ray_context().init()

        self._validate()

        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()
        validate_type(y, "y", pd.Series)
        validate_type(X, "X", pd.DataFrame)

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same lengths, got X: {len(X)}, y: {len(y)}"
            )

        self.X_steps_ = []
        self.y_steps_ = []
        self.random_seed_ = self.random_state or np.random.randint(0, 10000)

        problem_type = self._translate_problem_type(self.problem_type)

        self.level_ = ComponentLevel.translate(self.level)

        X = X.copy()
        y = y.copy()

        y = clean_df(y)
        self._drop_y_nan_rows(X, y)

        logger.info("Preparing DataFrames")

        y_validator = PrepareDataFrame(
            float_dtype=self.float_dtype, int_dtype=self.int_dtype, copy_X=False
        )
        X_validator = PrepareDataFrame(
            float_dtype=self.float_dtype,
            int_dtype=self.int_dtype,
            copy_X=False,
            ordinal_columns=ordinal_columns,
        )

        y = y_validator.fit_transform(y)
        X = X_validator.fit_transform(X)

        y_validator.set_params(copy_X=True)
        X_validator.set_params(copy_X=True)

        self.problem_type_, y = self._determine_problem_type(problem_type, y)

        self.cv_ = get_cv_for_problem_type(self.problem_type_, self.cv)
        self.stacking_cv_ = get_cv_for_problem_type(self.problem_type_, self.stacking_cv)

        y.index = list(X.index)

        self.y_steps_.append(("PrepareTarget", y_validator))
        self.X_steps_.append(("PrepareData", X_validator))

        if self.problem_type_.is_classification():
            logger.info("Encoding labels in y")
            y_encoder = LabelEncoder()()
            y = y_encoder.fit_transform(y)
            self.y_steps_.append(("LabelEncoding", y_encoder))

        if X_test is not None and y_test is not None:
            logger.info("Using predefined test sets")
            self.X_test_ = X_validator.transform(X_test)
            self.y_test_ = y_validator.transform(y_test)
            X, y = self._shuffle_data(X, y)
        elif X_test is not None or y_test is not None:
            raise ValueError(
                f"When passing either X_test or y_test, the other parameter must not be None as well, got X_test: {type(X_test)}, y_test: {type(y_test)}"
            )
        elif self.test_size <= 0:
            logger.info("test_size <= 0, no holdout testing will be performed")
            self.X_test_ = None
            self.y_test_ = None
            X, y = self._shuffle_data(X, y)
        else:
            logger.info(
                f"Splitting data into training and test sets ({1-(self.test_size*100)}-{(self.test_size*100)})"
            )
            X, self.X_test_, y, self.y_test_ = self._make_test_split(X, y)

        categorical_columns = X.dtypes.apply(lambda x: DataType.is_categorical(x))
        numeric_columns = list(categorical_columns[~categorical_columns].index)
        categorical_columns = list(categorical_columns[categorical_columns].index)

        self.trainer_ = Trainer(
            problem_type=self.problem_type_,
            random_state=self.random_seed_,
            level=self.level_,
            cv=self.cv_,
            stacking_cv=self.stacking_cv_,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            target_metric=self.target_metric,
            **self.trainer_config
            # cache="/home/baum/Documents/Coding/Python/automl",
        )

        self.X_ = X
        self.y_ = y

        self.trainer_.fit(X, y, X_test=self.X_test_, y_test=self.y_test_)

        self.results_ = self._get_results()
        self.best_id_ = str(self.results_.index[0])
        # self._displays["results_display"].clear_all()
        self._displays["results_display"].display(self.results_.head(20))

        return self

    @property
    def best_pipeline_(self):
        check_is_fitted(self)
        return self.get_pipeline_by_id(self.best_id_)

    def visualize_pipeline(self, id_or_pipeline: Union[str, Pipeline]):
        if isinstance(id_or_pipeline, Pipeline):
            pipeline = id_or_pipeline
        else:
            pipeline = self._get_pipeline_by_id(id_or_pipeline, refit=False, copy=False)
        html_repr = HTML(estimator_html_repr(pipeline))
        # self._displays["pipeline_display"].display(html_repr, display_type="html")
        return html_repr

    def get_pipeline_by_id(self, id, refit: Union[bool, str] = False):
        return self._get_pipeline_by_id(id, refit=refit, copy=True)

    def _get_pipeline_by_id(
        self, id, refit: Union[bool, str] = False, copy: bool = True
    ):
        check_is_fitted(self)
        pipeline = (
            deepcopy(self.trainer_.get_ensemble_or_pipeline_by_id(id))
            if copy
            else self.trainer_.get_ensemble_or_pipeline_by_id(id)
        )
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(steps=[("Ensemble", pipeline)])
        if refit:
            with joblib.parallel_backend("ray"):
                if refit == "on_test":
                    pipeline.fit(
                        pd.concat((self.X_, self.X_test_)),
                        pd.concat((self.y_, self.y_test_)),
                    )
                else:
                    pipeline.fit(self.X_, self.y_)
        pipeline.steps = self.X_steps_ + pipeline.steps
        return pipeline

    def _get_component_name(self, component):
        if component == "passthrough":
            return None
        r = None
        if isinstance(component, Pipeline):
            r = [
                self._get_component_name(subcomponent)
                for name, subcomponent in component.steps
            ]
        elif isinstance(component, ColumnTransformer):
            r = [
                self._get_component_name(subcomponent)
                for name, subcomponent, columns in component.transformers
            ]
        if r is not None:
            r = [x for x in r if x]
            return list(flatten_iterable(r)) if r else None
        return get_obj_name(component)

    def _get_result(self, result, stacking_level: int = 0):
        result = result.copy()
        component_names = [
            self._get_component_name(component)
            for name, component in result["estimator"].steps[:-1]
        ]
        component_names = [x for x in component_names if x]
        component_names = flatten_iterable(component_names)
        d = {
            "Id": result["trial_id"],
            "Pipeline": (
                f"{get_obj_name(result['estimator'].steps[-1][1])} w/ "
                f"{', '.join(component_names)}"
            ),
        }

        d["Stacking Level"] = stacking_level

        if "test_metrics" in result:
            d.update(
                {
                    f"Test {key}": metric
                    for key, metric in result["test_metrics"].items()
                }
            )
        if "metrics" in result:
            d.update(
                {
                    f"Validation {key}": metric
                    for key, metric in result["metrics"].items()
                }
            )

        d["Total Time (s)"] = result["time_total_s"]
        d["Estimator Fit Time (s)"] = result["estimator_fit_time"]
        d["Dataset Fraction"] = result.get("dataset_fraction", 1.0)

        return d

    # TODO unify with above
    def _get_ensemble_result(
        self, ensemble, ensemble_name: str, test_metrics: dict, stacking_level: int = 0
    ):
        d = {
            "Id": f"{stacking_level}_{ensemble_name}",
            "Pipeline": get_obj_name(ensemble),
        }

        d["Stacking Level"] = stacking_level

        d.update({f"Test {key}": metric for key, metric in test_metrics.items()})
        d.update({f"Validation {key}": None for key, metric in test_metrics.items()})

        d["Total Time (s)"] = None
        d["Estimator Fit Time (s)"] = None
        d["Dataset Fraction"] = 1.0

        return d

    def _get_results(self, show_full_trials_only: bool = True):
        check_is_fitted(self)
        all_results = []
        for stacking_level, results in enumerate(self.trainer_.all_results_):
            all_results.extend(
                [
                    self._get_result(result, stacking_level)
                    for trial_id, result in results.items()
                    if not show_full_trials_only
                    or result.get("dataset_fraction", 1.0) >= 1.0
                ]
            )
        for stacking_level, ensembles in enumerate(self.trainer_.ensembles_):
            all_results.extend(
                [
                    self._get_ensemble_result(
                        ensemble,
                        ensemble_name,
                        self.trainer_.ensemble_results_[stacking_level][ensemble_name],
                        stacking_level,
                    )
                    for ensemble_name, ensemble in ensembles.items()
                ]
            )

        metric_to_sort_by = (
            self.trainer_.target_metric or self.trainer_.default_metric_name
        )
        df = (
            pd.DataFrame(all_results)
            .set_index("Id", drop=True)
            .sort_values(
                by=[f"Test {metric_to_sort_by}", f"Validation {metric_to_sort_by}"],
                ascending=False,
            )
        )
        return df
