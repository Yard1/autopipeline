import numpy as np

from .transformer import DataType


def categorical_column_to_int(col, int_dtype: type = np.int32):
    if DataType.is_categorical(col.dtype):
        col.cat.categories = np.arange(
            start=0, stop=len(col.cat.categories), dtype=int_dtype
        )
    return col
