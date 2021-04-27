from enum import Enum

from pandas.api.types import (
    is_numeric_dtype,
    is_categorical_dtype,
)


class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"

    @staticmethod
    def is_numeric(dtype: type) -> bool:
        return is_numeric_dtype(dtype)

    @staticmethod
    def is_categorical(dtype: type) -> bool:
        return is_categorical_dtype(dtype)
