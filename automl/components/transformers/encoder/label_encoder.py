from .encoder import Encoder
from ..transformer import DataType
from ...component import ComponentLevel
from ....problems import ProblemType

from automl_models.components.transformers.encoder.label_encoder import (
    PandasLabelEncoder,
)


class LabelEncoder(Encoder):
    _component_class = PandasLabelEncoder
    _allowed_dtypes = {DataType.CATEGORICAL}
    _problem_types = {
        ProblemType.BINARY,
        ProblemType.MULTICLASS,
    }
    _component_level = ComponentLevel.NECESSARY
