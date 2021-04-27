from .imblearn import ImblearnSampler
from ...component import ComponentLevel
from ....problems import ProblemType
from ....search.distributions import IntUniformDistribution

from automl_models.components.transformers.imblearn.smote import PandasAutoSMOTE


class AutoSMOTE(ImblearnSampler):
    _component_class = PandasAutoSMOTE
    _default_parameters = {
        "k_neighbors": 5,
        "sampling_strategy": "auto",
        "random_state": 0,
        "n_jobs": None,
    }
    _component_level = ComponentLevel.UNCOMMON
    _problem_types = {ProblemType.BINARY, ProblemType.MULTICLASS}

    _default_tuning_grid = {
        "k_neighbors": IntUniformDistribution(2, 20, log=True, cost_related=False),
    }
