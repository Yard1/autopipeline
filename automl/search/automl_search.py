from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .utils import create_pipeline_blueprint, convert_tuning_grid
from ..components import DataType
from ..problems import ProblemType
from ..components import PrepareDataFrame, clean_df, LabelEncoder
from ..utils import validate_type

import warnings


class AutoML(BaseEstimator):
    def __init__(
        self,
        problem_type: Optional[Union[str, ProblemType]] = None,
        validation_size: float = 0.2,
        random_state: Optional[int] = None, # TODO: support other random states
        float_dtype: type = np.float32,
        int_dtype: Optional[type] = None,
    ) -> None:
        self.validation_size = validation_size
        self.problem_type = problem_type
        self.random_state = random_state
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        super().__init__()

        self._validate()

    def _validate(self):
        validate_type(self.problem_type, "problem_type", (str, ProblemType, type(None)))
        validate_type(self.validation_size, "validation_size", float)
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
        warnings.warn(f"Dropping {len(nan_ilocs)} y rows with NaNs: {list(nan_ilocs)}")
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

        if DataType.is_numeric(y.dtype):
            determined_problem_type = ProblemType.REGRESSION
        elif DataType.is_categorical(y.dtype):
            determined_problem_type = (
                ProblemType.BINARY
                if len(y.cat.categories) == 2
                else ProblemType.MULTICLASS
            )
        elif problem_type is None:
            raise ValueError("Could not determine problem type.")
        else:
            warnings.warn("Could not determine problem type.")

        if (
            determined_problem_type is not None
            and problem_type is not None
            and determined_problem_type != problem_type
        ):
            warnings.warn(
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

    def _make_validation_split(self, X, y):
        return train_test_split(X, y, test_size=self.validation_size, random_state=self.random_seed_, stratify=y if self.problem_type_.is_classification() else None)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_validation:Optional[pd.DataFrame]=None, y_validation:Optional[pd.Series]=None):
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
        self.random_seed_ = self.random_state

        problem_type = self._translate_problem_type(self.problem_type)

        X = X.copy()
        y = y.copy()

        y = clean_df(y)
        self._drop_y_nan_rows(X, y)

        y_validator = PrepareDataFrame(
            float_dtype=self.float_dtype, int_dtype=self.int_dtype, copy_X=False
        )
        X_validator = PrepareDataFrame(
            float_dtype=self.float_dtype, int_dtype=self.int_dtype, copy_X=False
        )

        y = y_validator.fit_transform(y)
        X = X_validator.fit_transform(X)

        y_validator.set_params(copy_X=True)
        X_validator.set_params(copy_X=True)

        self.problem_type_, y = self._determine_problem_type(problem_type, y)

        y.index = list(X.index)

        self.y_steps_.append(y_validator)
        self.X_steps_.append(X_validator)

        if self.problem_type_.is_classification():
            y_encoder = LabelEncoder()()
            y = y_encoder.fit_transform(y)
            self.y_steps_.append(y_encoder)

        if X_validation is not None and y_validation is not None:
            self.X_validation_ = X_validator.transform(X_validation)
            self.y_validation_ = y_validator.transform(y_validation)
        elif X_validation is not None or y_validation is not None:
            raise ValueError(f"When passing either X_validation or y_validation, the other parameter must not be None as well, got X_validation: {type(X_validation)}, y_validation: {type(y_validation)}")
        elif self.validation_size <= 0:
            self.X_validation_ = None
            self.y_validation_ = None
        else:
            X, self.X_validation_, y, self.y_validation_ = self._make_validation_split(X, y)

        return X, y