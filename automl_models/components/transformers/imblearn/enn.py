from imblearn.under_sampling import EditedNearestNeighbours as _EditedNearestNeighbours

from ...compatibility.pandas import PandasDataFrameTransformerMixin


class PandasEditedNearestNeighbours(PandasDataFrameTransformerMixin, _EditedNearestNeighbours):
    pass
