from ....search.distributions import CategoricalDistribution
from ...transformers.utils import categorical_column_to_int_categories


def _estimate_gamma(X, default_value) -> CategoricalDistribution:
    gamma_values = [default_value]
    if X is not None:
        num_features = X.shape[1]
        gamma_values.append(float(1 / num_features))
        X = X.apply(categorical_column_to_int_categories)
        if hasattr(X, "values"):
            X = X.values
        X_var = X.var()
        if X_var != 0:
            gamma_values.append(float(1 / (num_features * X_var)))
    return CategoricalDistribution(reversed(gamma_values))


def estimate_gamma_nystroem(config, stage) -> CategoricalDistribution:
    X = config.X
    return _estimate_gamma(X, 0.1)


def estimate_gamma_pcs(config, stage) -> CategoricalDistribution:
    X = config.X
    return _estimate_gamma(X, 1)