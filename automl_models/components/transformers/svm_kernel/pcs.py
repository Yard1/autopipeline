from sklearn.kernel_approximation import PolynomialCountSketch as _PolynomialCountSketch
from sklearn.preprocessing import Normalizer

from .utils import GammaMixin


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
