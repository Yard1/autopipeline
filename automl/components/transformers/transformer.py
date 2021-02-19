from enum import Enum
from ..component import Component

from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_datetime64_any_dtype,
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


class Transformer(Component):
    _allowed_dtypes = {
        DataType.NUMERIC,
        DataType.CATEGORICAL,
    }
