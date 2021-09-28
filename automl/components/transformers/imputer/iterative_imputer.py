from .imputer import Imputer
from ...component import ComponentLevel

from automl_models.components.transformers.imputer.iterative_imputer import (
    PandasIterativeImputer,
)


class IterativeImputer(Imputer):
    _component_class = PandasIterativeImputer
    _default_parameters = {
        "regressor": "LGBMRegressor",
        "classifier": "LGBMClassifier",
        "max_iter": 4,
        "verbose": 0,
        "random_state": 0,
        "n_jobs": 1,
    }
    _component_level = ComponentLevel.UNCOMMON


# class SimpleCategoricalImputer(Imputer):
#     _component_class = PandasSimpleCategoricalImputer
#     _default_parameters = {
#         "strategy": "most_frequent",
#         "fill_value": "missing_value",
#         "verbose": 0,
#         "copy": True,
#         "add_indicator": False,
#     }
#     _default_tuning_grid = {
#         "strategy": CategoricalDistribution(["most_frequent", "constant"])
#     }
#     _allowed_dtypes = {DataType.CATEGORICAL}
#     _component_level = ComponentLevel.NECESSARY
