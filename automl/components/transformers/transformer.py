from enum import Enum
from ..component import Component


class DataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class Transformer(Component):
    _allowed_dtypes = {
        DataType.NUMERICAL,
        DataType.CATEGORICAL,
    }
