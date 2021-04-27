from sklearn.kernel_approximation import Nystroem

from .utils import GammaMixin


class NystroemDynamicGamma(GammaMixin, Nystroem):
    pass
