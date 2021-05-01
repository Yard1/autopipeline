from typing import Any, Tuple, Union
from sklearn.base import clone


def clone_with_n_jobs_1(estimator, *, safe: bool = True):
    estimator = clone(estimator, safe=safe)
    params = estimator.get_params()
    params_to_set = {param: 1 for param in params.keys() if param.endswith("n_jobs")}
    estimator.set_params(**params_to_set)
    # clone twice to deal with nested
    estimator = clone(estimator, safe=safe)
    return estimator


def validate_type(var: Any, var_name: str, types: Union[type, Tuple[type]]):
    if types is None and var is None:
        return
    if isinstance(types, tuple) and None in types and var is None:
        return
    if not isinstance(var, types):
        raise TypeError(f"Expected {var_name} to be of type {types}, got {type(var)}")


def removeprefix(string: str, prefix: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix) :]
    else:
        return string[:]


def removesuffix(string: str, suffix: str) -> str:
    if suffix and string.endswith(suffix):
        return string[: -len(suffix)]
    else:
        return string[:]