import pandas as pd

from imblearn.under_sampling import EditedNearestNeighbours as _EditedNearestNeighbours
from sklearn.base import BaseEstimator, clone

from ...compatibility.pandas import *
from .imblearn import ImblearnSampler
from ..transformer import DataType
from ...component import ComponentLevel
from ....problems import ProblemType


class PandasEditedNearestNeighbours(PandasDataFrameTransformerMixin, _EditedNearestNeighbours):
    pass