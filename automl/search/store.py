from typing import Any, Type
from fastcore.basics import num_cpus
import ray
import ray.cloudpickle as cpickle
from collections import defaultdict
import lz4.frame
from sklearn.utils.validation import check_is_fitted, NotFittedError


def compress(value, **kwargs):
    if "compression_level" not in kwargs:
        kwargs["compression_level"] = 9
    return lz4.frame.compress(cpickle.dumps(value), **kwargs)


def decompress(value):
    return cpickle.loads(lz4.frame.decompress(value))


class CachedObject:
    def __init__(self, store_actor_name, store_name: str, key: str) -> None:
        self.store_actor_name = store_actor_name
        self.store_name = store_name
        self.key = key

    def _set_object(self, obj: Any):
        self.obj_type = type(obj)
        ray.get(self.store_actor.put.remote(self.key, self.store_name, obj))

    def _get_object(self):
        ret = ray.get(self.store_actor.get.remote(self.key, self.store_name))
        self.obj_type = type(ret)
        return ret

    @property
    def store_actor(self):
        if not hasattr(self, "_store_actor") or self._store_actor is None:
            try:
                self._store_actor = ray.get_actor(self.store_actor_name)
            except Exception:
                self._store_actor = None
        return self._store_actor

    @property
    def object(self) -> Any:
        return self._get_object()

    @object.setter
    def object(self, obj: Any):
        return self._set_object(obj)

    def __getstate__(self):
        ret = self.__dict__.copy()
        ret.pop("_store_actor", None)
        return ret


class CachedSklearn(CachedObject):
    def fit(self, X, y=None):
        pass

    def _get_object(self):
        ret = super()._get_object()
        ret._ray_cached_object = self
        return ret

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "fitted_")

    @is_fitted.setter
    def is_fitted(self, val: bool):
        if val:
            self.fitted_ = True
        elif hasattr(self, "fitted_"):
            del self.fitted_


def cached_object_factory(obj, store_actor, store_name, key) -> CachedObject:
    if hasattr(obj, "fit"):
        ret = CachedSklearn(store_actor.name, store_name, key)
        try:
            check_is_fitted(obj)
            ret.is_fitted = True
        except NotFittedError:
            pass
    else:
        ret = CachedObject(store_actor.name, store_name, key)
    ret.obj_type = type(obj)
    return ret


@ray.remote(max_restarts=20, num_cpus=0)
class RayStore(object):
    def __init__(self, name: str) -> None:
        self.values = defaultdict(dict)
        self.name = name

    def get(self, key, store_name, pop=False):
        if pop:
            v = self.values[store_name].pop(key)
        else:
            v = self.values[store_name][key]
        return v

    def get_cached_object(self, key, store_name) -> CachedObject:
        v = self.values[store_name][key]
        v = cached_object_factory(v, self, store_name, key)
        return v

    def put(self, key, store_name, value):
        self.values[store_name][key] = value

    def get_all_keys(self, store_name) -> list:
        return list(self.values[store_name].keys())

    def get_all_refs(self, store_name, pop=False) -> list:
        r = [
            self.get(key, store_name, pop=pop) for key in self.get_all_keys(store_name)
        ]
        return r

    def get_all(self, store_name) -> dict:
        return self.values[store_name]

    def get_all_cached_objects(self, store_name) -> dict:
        return {
            key: self.get_cached_object(key, store_name)
            for key in self.values[store_name]
        }

    def get_values(self) -> dict:
        return self.values
