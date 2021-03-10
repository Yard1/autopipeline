import numpy as np

from .transformer import DataType

# TODO: look into codes
def categorical_column_to_int_categories(col, int_dtype: type = np.uint8):
    if DataType.is_categorical(col.dtype):
        col.cat.categories = np.arange(
            start=0, stop=len(col.cat.categories), dtype=int_dtype
        )
    return col
