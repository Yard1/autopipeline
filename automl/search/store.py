from typing import Any, Type
import gc
import ray
import ray.cloudpickle as cpickle
from collections import defaultdict
import lz4.frame
from sklearn.utils.validation import check_is_fitted, NotFittedError


# def compress(value, **kwargs):
#     if "compression_level" not in kwargs:
#         kwargs["compression_level"] = 9
#     return lz4.frame.compress(cpickle.dumps(value), **kwargs)


# def decompress(value):
#     return cpickle.loads(lz4.frame.decompress(value))


# class CompressedData:
#     def __init__(self, data):
#         self.data_type = type(data)
#         if hasattr(data, "fit"):
#             self.fit = None
#             try:
#                 check_is_fitted(data)
#                 self.is_fitted = True
#             except NotFittedError:
#                 pass
#         self.data = compress(data)

#     def get_decompressed_data(self):
#         return decompress(self.data)


# class WrappedRef:
#     def __init__(self, data):
#         if isinstance(data, CompressedData):
#             self.is_compressed = True
#             self.data_type = data.data_type
#         else:
#             self.is_compressed = False
#             self.data_type = type(data)
#         if hasattr(data, "fit"):
#             self.fit = None
#             try:
#                 check_is_fitted(data)
#                 self.is_fitted = True
#             except NotFittedError:
#                 self.is_fitted = False
#         self.ref = ray.put(data) if not isinstance(data, ray.ObjectRef) else data


# class CachedObject:
#     def __init__(
#         self, store_actor_name, store_name: str, key: str, compress: bool = False
#     ) -> None:
#         self.store_actor_name = store_actor_name
#         self.store_name = store_name
#         self.key = key
#         self.compress = compress

#     def _set_object(self, obj: Any):
#         self.obj_type = type(obj)
#         if self.compress and not isinstance(obj, CompressedData):
#             obj = CompressedData(obj)
#         self.store_actor.put.remote(
#             self.key, self.store_name, obj, compress=self.compress
#         )

#     def _get_object(self):
#         if not hasattr(self, "_object"):
#             ref: WrappedRef = ray.get(
#                 self.store_actor.get.remote(self.key, self.store_name)
#             )
#             ret = ray.get(ref.ref)
#             if isinstance(ret, CompressedData):
#                 ret = ret.get_decompressed_data()
#             self.obj_type = type(ret)
#             self._object = ret
#         return self._object

#     def clear_cache(self):
#         if hasattr(self, "_store_actor"):
#             del self._store_actor

#     @property
#     def store_actor(self):
#         if not hasattr(self, "_store_actor") or self._store_actor is None:
#             try:
#                 self._store_actor = ray.get_actor(self.store_actor_name)
#             except Exception:
#                 self._store_actor = None
#         return self._store_actor

#     @property
#     def object(self) -> Any:
#         return self._get_object()

#     @object.setter
#     def object(self, obj: Any):
#         return self._set_object(obj)

#     def __getstate__(self):
#         ret = self.__dict__.copy()
#         ret.pop("_store_actor", None)
#         return ret


# class CachedSklearn(CachedObject):
#     def fit(self, X, y=None):
#         pass

#     def _get_object(self):
#         ret = super()._get_object()
#         ret._ray_cached_object = self
#         return ret

#     @property
#     def is_fitted(self) -> bool:
#         return hasattr(self, "fitted_")

#     @is_fitted.setter
#     def is_fitted(self, val: bool):
#         if val:
#             self.fitted_ = True
#         elif hasattr(self, "fitted_"):
#             del self.fitted_


# def cached_object_factory(
#     obj: WrappedRef, store_actor, store_name, key, store_actor_handle=None
# ) -> CachedObject:
#     compress = obj.is_compressed
#     if hasattr(obj, "fit"):
#         ret = CachedSklearn(store_actor.name, store_name, key, compress=compress)
#         ret.is_fitted = obj.is_fitted
#     else:
#         ret = CachedObject(store_actor.name, store_name, key, compress=compress)
#     ret.obj_type = obj.data_type
#     if store_actor_handle:
#         ret._store_actor = store_actor_handle
#     return ret


# @ray.remote(max_restarts=20, num_cpus=0)
# class RayStore(object):
#     def __init__(self, name: str) -> None:
#         self.values = defaultdict(dict)
#         self.name = name
#         self.actor_handle = ray.get_actor(self.name)

#     def get(self, key, store_name, pop=False) -> WrappedRef:
#         if pop:
#             value = self.values[store_name].pop(key)
#         else:
#             value = self.values[store_name][key]
#         return value

#     def get_cached_object(self, key, store_name) -> CachedObject:
#         v = self.get(key, store_name)
#         v = cached_object_factory(
#             v, self, store_name, key, store_actor_handle=self.actor_handle
#         )
#         return v

#     def put(self, key, store_name, value, compress=False):
#         if isinstance(value, WrappedRef):
#             self.values[store_name][key] = value
#         else:
#             if compress and not isinstance(value, CompressedData):
#                 value = CompressedData(value)
#             self.values[store_name][key] = WrappedRef(value)
#         gc.collect()

#     def get_all_keys(self, store_name) -> list:
#         return list(self.values[store_name].keys())

#     def get_all_refs(self, store_name, pop=False, decompress=True) -> list:
#         r = [
#             self.get(key, store_name, pop=pop, decompress=decompress)
#             for key in self.get_all_keys(store_name)
#         ]
#         return r

#     def get_all(self, store_name) -> dict:
#         return self.values[store_name]

#     def get_all_cached_objects(self, store_name) -> dict:
#         return {
#             key: self.get_cached_object(key, store_name)
#             for key in self.values[store_name]
#         }

#     def get_values(self) -> dict:
#         return self.values

#     def clean(self, store_name):
#         del self.values[store_name]
