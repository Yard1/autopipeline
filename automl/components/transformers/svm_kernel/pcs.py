from sklearn.kernel_approximation import PolynomialCountSketch as _PolynomialCountSketch
from sklearn.preprocessing import MinMaxScaler, Normalizer

from .utils import GammaMixin
from .svm_kernel import SVMKernel
from ..transformer import DataType
from ...component import ComponentLevel
from ...compatibility.pandas import PandasDataFrameTransformerMixin

from ....search.distributions import (
    CategoricalDistribution,
    UniformDistribution,
    IntUniformDistribution,
    UniformDistribution,
    FunctionDistribution,
)

from ...component import ComponentConfig
from ....search.stage import AutoMLStage
from ...estimators.linear_model.linear_model_estimator import LinearModelEstimator


class PolynomialCountSketchDynamicNComponents(GammaMixin, _PolynomialCountSketch):
    def fit(self, X, y=None):
        n_features = X.shape[1]
        self._n_components = self.n_components
        self.n_components = max(self.n_components, 10 * n_features)
        self.normalizer_ = Normalizer()
        r = super().fit(self.normalizer_.fit_transform(X), y=y)
        new_n_components = self.n_components
        self.n_components = self._n_components
        self._n_components = new_n_components
        return r

    def transform(self, X):
        old_n_components = self.n_components
        self.n_components = self._n_components
        r = super().transform(self.normalizer_.transform(X))
        self.n_components = old_n_components
        return r


class PolynomialCountSketch(SVMKernel):
    _component_class = PolynomialCountSketchDynamicNComponents
    _default_parameters = {
        "gamma": 1.0,
        "coef0": 0,
        "degree": 3,
        "n_components": 500,
        "random_state": 0,
    }
    _allowed_dtypes = {DataType.NUMERIC, DataType.CATEGORICAL}
    _component_level = ComponentLevel.RARE

    _default_tuning_grid = {
        "gamma": CategoricalDistribution(["scale", 1.0, "auto"]),
        #    "coef0": UniformDistribution(-1, 1),
        "degree": IntUniformDistribution(2, 3),
    }

    def is_component_valid(self, config: ComponentConfig, stage: AutoMLStage) -> bool:
        if config is None:
            return True
        super_check = super().is_component_valid(config, stage)
        return super_check and (
            config.estimator is None
            or isinstance(config.estimator, LinearModelEstimator)
        )