from typing import Any, Tuple, Union
from sklearn.base import clone
import ray


def clone_with_n_jobs(estimator, *, n_jobs: bool = 1, safe: bool = True):
    estimator = clone(estimator, safe=safe)
    params = estimator.get_params()
    params_to_set = {param: n_jobs for param in params.keys() if param.endswith("n_jobs") or param.endswith("thread_count")}
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


def split_list_into_chunks(lst: list, chunk_size: int) -> list:
    return [
        lst[i * chunk_size : (i + 1) * chunk_size]
        for i in range((len(lst) + chunk_size - 1) // chunk_size)
    ]


def ray_put_if_needed(obj: Any) -> ray.ObjectRef:
    if isinstance(obj, ray.ObjectRef):
        return obj
    return ray.put(obj)


def ray_get_if_needed(obj: ray.ObjectRef) -> Any:
    if isinstance(obj, ray.ObjectRef):
        return ray.get(obj)
    return obj
