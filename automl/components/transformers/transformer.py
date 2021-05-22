from ..component import Component

from automl_models.components.transformers.transformer import DataType


class Transformer(Component):
    _allowed_dtypes = {
        DataType.NUMERIC,
        DataType.CATEGORICAL,
    }
