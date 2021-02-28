from typing import Dict, Optional

from collections.abc import Hashable
import numpy as np
from copy import copy


class _OptunaParam:
    def __getattr__(self, item):
        def _inner(*args, **kwargs):
            return (item, args, kwargs)

        return _inner


optuna_param = _OptunaParam()


class Distribution:
    def __init__(self):
        raise NotImplementedError("This is an abstract class.")

    def get_skopt(self):
        raise NotImplementedError("This is an abstract class.")

    def get_optuna(self):
        raise NotImplementedError("This is an abstract class.")

    def get_optuna_dist(self):
        raise NotImplementedError("This is an abstract class.")

    def get_hyperopt(self, label):
        raise NotImplementedError("This is an abstract class.")

    def get_CS(self, label):
        raise NotImplementedError("This is an abstract class.")

    def get_tune(self):
        raise NotImplementedError("This is an abstract class.")

    def _validate_default(self, default):
        raise NotImplementedError("This is an abstract class.")

    @property
    def default(self):
        if not hasattr(self, "_default"):
            raise KeyError("default value has not been set.")
        return self._default

    @default.setter
    def default(self, val):
        self._validate_default(val)
        self._default = val


class UniformDistribution(Distribution):
    """
    Uniform float distribution.

    Parameters
    ----------
    lower: float
        Inclusive lower bound of distribution.
    upper: float
        Inclusive upper bound of distribution.
    log: bool, default = False:
        If True, the distribution will be log-uniform.
    """

    def __init__(self, lower: float, upper: float, log: bool = False):
        self.lower = lower
        self.upper = upper
        self.log = log

    def _validate_default(self, default):
        if not (self.lower <= default <= self.upper):
            raise ValueError(
                f"default value {default} must be between {self.lower} and {self.upper}"
            )

    def get_skopt(self):
        import skopt.space

        if self.log:
            return skopt.space.Real(self.lower, self.upper, prior="log-uniform")
        else:
            return skopt.space.Real(self.lower, self.upper, prior="uniform")

    def get_optuna(self, label):
        if self.log:
            return optuna_param.suggest_loguniform(label, self.lower, self.upper)
        else:
            return optuna_param.suggest_uniform(label, self.lower, self.upper)

    def get_optuna_dist(self):
        from optuna.distributions import LogUniformDistribution, UniformDistribution

        if self.log:
            return LogUniformDistribution(self.lower, self.upper)
        else:
            return UniformDistribution(self.lower, self.upper)

    def get_hyperopt(self, label):
        from hyperopt import hp

        if self.log:
            return hp.loguniform(label, np.log(self.lower), np.log(self.upper))
        else:
            return hp.uniform(label, self.lower, self.upper)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        try:
            return CSH.UniformFloatHyperparameter(
                name=label,
                lower=self.lower,
                upper=self.upper,
                log=self.log,
                default_value=self.default,
            )
        except KeyError:
            return CSH.UniformFloatHyperparameter(
                name=label,
                lower=self.lower,
                upper=self.upper,
                log=self.log,
            )

    def get_tune(self):
        from ray import tune

        if self.log:
            return tune.loguniform(lower=self.lower, upper=self.upper)
        else:
            return tune.uniform(lower=self.lower, upper=self.upper)

    def __repr__(self):
        return f"UniformDistribution(lower={self.lower}, upper={self.upper}, log={self.log})"


class IntUniformDistribution(Distribution):
    """
    Uniform integer distribution.

    Parameters
    ----------
    lower: int
        Inclusive lower bound of distribution.
    upper: int
        Inclusive upper bound of distribution.
    log: bool, default = False:
        If True, the distribution will be log-uniform.
    """

    def __init__(self, lower: int, upper: int, log: bool = False):
        self.lower = lower
        self.upper = upper
        self.log = log

    def _validate_default(self, default):
        if not (self.lower <= default <= self.upper):
            raise ValueError(
                f"default value {default} must be between {self.lower} and {self.upper}"
            )

    def get_skopt(self):
        import skopt.space

        if self.log:
            return skopt.space.Integer(self.lower, self.upper, prior="log-uniform")
        else:
            return skopt.space.Integer(self.lower, self.upper, prior="uniform")

    def get_optuna(self, label):
        if self.log:
            return optuna_param.suggest_int(label, self.lower, self.upper, log=True)
        else:
            return optuna_param.suggest_int(label, self.lower, self.upper, log=False)

    def get_optuna_dist(self):
        from optuna.distributions import (
            IntLogUniformDistribution,
            IntUniformDistribution,
        )

        if self.log:
            return IntLogUniformDistribution(self.lower, self.upper)
        else:
            return IntUniformDistribution(self.lower, self.upper)

    def get_hyperopt(self, label):
        from hyperopt import hp
        from hyperopt.pyll import scope

        if self.log:
            return scope.int(
                hp.qloguniform(label, np.log(self.lower), np.log(self.upper), 1)
            )
        else:
            return scope.int(hp.quniform(label, self.lower, self.upper, 1))

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        try:
            return CSH.UniformIntegerHyperparameter(
                name=label,
                lower=self.lower,
                upper=self.upper,
                log=self.log,
                default_value=self.default,
            )
        except KeyError:
            return CSH.UniformIntegerHyperparameter(
                name=label,
                lower=self.lower,
                upper=self.upper,
                log=self.log,
            )

    def get_tune(self):
        from ray import tune

        if self.log:
            return tune.lograndint(self.lower, self.upper)
        else:
            return tune.randint(self.lower, self.upper)

    def __repr__(self):
        return f"IntUniformDistribution(lower={self.lower}, upper={self.upper}, log={self.log})"


class DiscreteUniformDistribution(Distribution):
    """
    Discrete (with step) uniform float distribution.

    Parameters
    ----------
    lower: float
        Inclusive lower bound of distribution.
    upper: float
        Inclusive upper bound of distribution.
    q: float = None:
        Step. If None, will be equal to UniformDistribution.

    Warnings
    --------
    - Due to scikit-optimize not supporting discrete distributions,
    `get_skopt()` will return a standard uniform distribution.
    """

    def __init__(self, lower: int, upper: int, q: Optional[float] = None):
        self.lower = lower
        self.upper = upper
        self.q = q

    def _validate_default(self, default):
        if not (self.lower <= default <= self.upper):
            raise ValueError(
                f"default value {default} must be between {self.lower} and {self.upper}"
            )

    def get_skopt(self):
        import skopt.space

        # not supported, return standard uniform distribution
        return skopt.space.Real(self.lower, self.upper, prior="uniform")

    def get_optuna(self, label):
        return optuna_param.suggest_discrete_uniform(
            label, self.lower, self.upper, self.q
        )

    def get_optuna_dist(self):
        from optuna.distributions import DiscreteUniformDistribution

        return DiscreteUniformDistribution(self.lower, self.upper, step=self.q)

    def get_hyperopt(self, label):
        from hyperopt import hp

        return hp.quniform(label, self.lower, self.upper, self.q)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        try:
            return CSH.UniformFloatHyperparameter(
                name=label,
                lower=self.lower,
                upper=self.upper,
                q=self.q,
                default_value=self.default,
            )
        except KeyError:
            return CSH.UniformFloatHyperparameter(
                name=label,
                lower=self.lower,
                upper=self.upper,
                q=self.q,
                default=self.default,
            )

    def get_tune(self):
        from ray import tune

        return tune.quniform(lower=self.lower, upper=self.upper, q=self.q)

    def __repr__(self):
        return f"DiscreteUniformDistribution(lower={self.lower}, upper={self.upper}, q={self.q})"


class CategoricalDistribution(Distribution):
    """
    Categorical distribution.

    Parameters
    ----------
    values: list or other iterable
        Possible values.

    Warnings
    --------
    - `None` is not supported  as a value for ConfigSpace.
    """

    None_str = "!None"

    def __init__(self, values):
        self.values = [x if x is not None else self.None_str for x in values]
        try:
            self.values = list(set(self.values))
        except:
            new_values = []
            for x in self.values:
                if x not in new_values:
                    new_values.append(x)
            self.values = new_values

    @property
    def default(self):
        if not hasattr(self, "_default"):
            raise KeyError("default value has not been set.")
        return self._default

    @default.setter
    def default(self, val):
        if val is None:
            val = self.None_str
        self._validate_default(val)
        self._default = val

    def _validate_default(self, default):
        if not default in self.values:
            raise ValueError(f"default value {default} must be in values {self.values}")

    def get_skopt(self):
        import skopt.space

        return skopt.space.Categorical(
            [x if isinstance(x, Hashable) else None for x in self.values],
            transform="identity",
        )

    def get_optuna(self, label):
        return optuna_param.suggest_categorical(label, self.values)

    def get_optuna_dist(self):
        from optuna.distributions import CategoricalDistribution

        return CategoricalDistribution(self.values)

    def get_hyperopt(self, label):
        from hyperopt import hp

        return hp.choice(label, self.values)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        try:
            return CSH.CategoricalHyperparameter(
                name=label,
                choices=self.values,
                default_value=self.default
                if self.default is not None
                else self.None_str,
            )
        except KeyError:
            return CSH.CategoricalHyperparameter(name=label, choices=self.values)

    def get_tune(self):
        from ray import tune

        return tune.choice(self.values)

    def __repr__(self):
        return f"CategoricalDistribution(values={self.values})"


class FunctionDistribution(Distribution):
    def __init__(self, function):
        self.function = function

    def _validate_default(self, default):
        return True

    def __call__(self, config, stage):
        dist = self.function(config, stage)
        try:
            dist.default = self.default
        except KeyError:
            pass
        return dist


def get_skopt_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_skopt() for k, v in distributions.items()}


def get_optuna_trial_suggestions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_optuna(k) for k, v in distributions.items()}


def get_optuna_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_optuna_dist() for k, v in distributions.items()}


def get_hyperopt_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_hyperopt(k) for k, v in distributions.items()}


def get_CS_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_CS(k) for k, v in distributions.items()}


def get_tune_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_tune() for k, v in distributions.items()}


def get_min_max(o):
    if isinstance(o, CategoricalDistribution):
        o = o.values
    elif isinstance(o, Distribution):
        return (o.lower, o.upper)

    o = sorted(o)
    return (o[0], o[-1])
