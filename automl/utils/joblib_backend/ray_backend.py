import gc
from pickle import PicklingError
import ray
from ray.util.joblib.ray_backend import RayBackend, PicklingPool
from ray.util.multiprocessing.pool import Pool, PoolActor
from ray.tune.registry import _ParameterRegistry
from joblib.parallel import register_parallel_backend
from joblib._parallel_backends import SafeFunction
from copy import copy
from pandas.api.types import is_scalar, is_bool

import logging

from ..memory.hashing import hash as joblib_hash

from pandas.api.types import is_array_like

logger = logging.getLogger(__name__)


def _is_obj_primitive(obj):
    return is_scalar(obj) or is_bool(obj)


def _can_cache(obj):
    return not _is_obj_primitive(obj) and is_array_like(obj)


class MultiprocessingCache(_ParameterRegistry):
    def is_in_cache(self, k: str):
        if ray.is_initialized():
            return k in self.references
        return k in self.to_flush

    def get_reference(self, k: str):
        if not ray.is_initialized():
            return self.to_flush[k]
        return self.references[k]

    def flush(self):
        for k, v in self.to_flush.items():
            if k in self.references:
                continue
            self.references[k] = ray.put(v)
        self.to_flush.clear()

    def clear(self):
        self.to_flush.clear()
        self.references.clear()
        gc.collect()


multiprocessing_cache = MultiprocessingCache()


class CachingPool(Pool):
    def _cache_func_items(self, func):
        original_func = func
        if isinstance(func, SafeFunction):
            func = copy(func.func)
        func.items = [
            self._convert_func_tuple_to_refs(func_tuple) for func_tuple in func.items
        ]
        return original_func

    def _convert_func_tuple_to_refs(self, func_tuple):
        func, args, kwargs = func_tuple
        if args:
            args = tuple([self._get_object_ref_from_cache(arg) for arg in args])
        if kwargs:
            kwargs = {
                k: self._get_object_ref_from_cache(arg) for k, arg in kwargs.items()
            }
        return func, args, kwargs

    def _run_batch(self, actor_index, func, batch):
        func = self._cache_func_items(func)
        return super()._run_batch(actor_index, func, batch)

    def _get_object_ref_from_cache(self, obj):
        if not _can_cache(obj):
            return obj
        try:
            obj_hash = joblib_hash(obj)
        except PicklingError:
            return obj
        if not multiprocessing_cache.is_in_cache(obj_hash):
            print(f"putting {obj_hash} ({type(obj)}) in multiprocessing cache")
            multiprocessing_cache.put(obj_hash, obj)
        print(f"accessing {obj_hash} ({type(obj)}) from multiprocessing cache")
        return multiprocessing_cache.get_reference(obj_hash)


class CachingRayBackend(RayBackend):
    def configure(
        self, n_jobs=1, parallel=None, prefer=None, require=None, **memmappingpool_args
    ):
        """Make Ray Pool the father class of PicklingPool. PicklingPool is a
        father class that inherits Pool from multiprocessing.pool. The next
        line is a patch, which changes the inheritance of Pool to be from
        ray.util.multiprocessing.pool.
        """
        PicklingPool.__bases__ = (CachingPool,)
        """Use all available resources when n_jobs == -1. Must set RAY_ADDRESS
        variable in the environment or run ray.init(address=..) to run on
        multiple nodes.
        """
        if n_jobs == -1:
            if not ray.is_initialized():
                import os

                if "RAY_ADDRESS" in os.environ:
                    logger.info(
                        "Connecting to ray cluster at address='{}'".format(
                            os.environ["RAY_ADDRESS"]
                        )
                    )
                else:
                    logger.info("Starting local ray cluster")
                ray.init()
            ray_cpus = int(ray.state.cluster_resources()["CPU"])
            n_jobs = ray_cpus

        eff_n_jobs = super(RayBackend, self).configure(
            n_jobs, parallel, prefer, require, **memmappingpool_args
        )
        return eff_n_jobs


def register_ray_caching():
    register_parallel_backend("ray_caching", CachingRayBackend)