from joblib import Memory
from joblib.memory import (
    MemorizedFunc,
    NotMemorizedFunc,
    MemorizedResult,
    NotMemorizedResult,
    _format_load_msg,
)
import functools
import time
import traceback
import tempfile
import os
import sys
import io

import numpy as np

import pickle
from joblib import hashing
from joblib.func_inspect import get_func_code, get_func_name, filter_args
from joblib.func_inspect import format_call
from joblib.func_inspect import format_signature
from joblib.logger import Logger, format_time, pformat
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.hashing import Hasher, NumpyHasher

from pandas.util import hash_pandas_object, hash_array
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndexClass,
    ABCMultiIndex,
    ABCSeries,
)

from xxhash import xxh3_128

Pickler = pickle._Pickler


class xxHasher(Hasher):
    def __init__(self, hash_name="md5"):
        self.stream = io.BytesIO()
        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        protocol = pickle.HIGHEST_PROTOCOL
        Pickler.__init__(self, self.stream, protocol=protocol)
        # Initialise the hash obj
        self._hash = xxh3_128()


class xxNumpyHasher(NumpyHasher):
    def __init__(self, hash_name="md5", coerce_mmap=False):
        """
        Parameters
        ----------
        hash_name: string
            The hash algorithm to be used
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
            objects.
        """
        self.coerce_mmap = coerce_mmap
        xxHasher.__init__(self, hash_name=hash_name)
        # delayed import of numpy, to avoid tight coupling
        import numpy as np

        self.np = np
        if hasattr(np, "getbuffer"):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview


class xxPandasHasher(xxNumpyHasher):
    def _hash_pandas(self, obj):
        try:
            hashed_obj = hash_pandas_object(obj)
            return (obj.__class__, hashed_obj._values, hashed_obj.index._values)
        except TypeError:
            return obj

    def hash(self, obj, return_digest=True):
        if isinstance(obj, dict):
            obj = (obj.__class__, {k: self._hash_pandas(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            obj = (obj.__class__, [self._hash_pandas(v) for v in obj])
        elif isinstance(obj, tuple):
            obj = (obj.__class__, tuple([self._hash_pandas(v) for v in obj]))
        else:
            obj = self._hash_pandas(obj)
        try:
            self.dump(obj)
        except pickle.PicklingError as e:
            e.args += ("PicklingError while hashing %r: %r" % (obj, e),)
            raise
        dumps = self.stream.getvalue()
        self._hash.update(dumps)
        if return_digest:
            return self._hash.hexdigest()

    def save(self, obj):
        """Subclass the save method, to hash ndarray subclass, rather
        than pickling them. Off course, this is a total abuse of
        the Pickler class.
        """
        if isinstance(obj, self.np.ndarray) and not (
            obj.dtype.hasobject and obj.ndim > 2
        ):
            # Compute a hash of the object
            # The update function of the hash requires a c_contiguous buffer.
            if obj.shape == ():
                # 0d arrays need to be flattened because viewing them as bytes
                # raises a ValueError exception.
                obj_c_contiguous = obj.flatten()
            elif obj.flags.c_contiguous:
                obj_c_contiguous = obj
            elif obj.flags.f_contiguous:
                obj_c_contiguous = obj.T
            else:
                # Cater for non-single-segment arrays: this creates a
                # copy, and thus aleviates this issue.
                # XXX: There might be a more efficient way of doing this
                obj_c_contiguous = obj.flatten()

            # If we have objects, we can use pandas's
            # fast hashing to turn them to ints
            if obj.dtype.hasobject:
                if obj_c_contiguous.ndim == 1:
                    obj_c_contiguous = hash_array(obj_c_contiguous)
                else:
                    hashes = []
                    for x in obj_c_contiguous:
                        hashes.append(hash_array(x))
                    obj_c_contiguous = np.array(hashes)

            # memoryview is not supported for some dtypes, e.g. datetime64, see
            # https://github.com/numpy/numpy/issues/4983. The
            # workaround is to view the array as bytes before
            # taking the memoryview.
            self._hash.update(self._getbuffer(obj_c_contiguous.view(self.np.uint8)))

            # We store the class, to be able to distinguish between
            # Objects with the same binary content, but different
            # classes.
            if self.coerce_mmap and isinstance(obj, self.np.memmap):
                # We don't make the difference between memmap and
                # normal ndarrays, to be able to reload previously
                # computed results with memmap.
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            # We also return the dtype and the shape, to distinguish
            # different views on the same data with different dtypes.

            # The object will be pickled by the pickler hashed at the end.
            obj = (
                klass,
                ("HASHED", obj.dtype, obj.shape, obj.strides),
            )
        elif isinstance(obj, self.np.dtype):
            # numpy.dtype consistent hashing is tricky to get right. This comes
            # from the fact that atomic np.dtype objects are interned:
            # ``np.dtype('f4') is np.dtype('f4')``. The situation is
            # complicated by the fact that this interning does not resist a
            # simple pickle.load/dump roundtrip:
            # ``pickle.loads(pickle.dumps(np.dtype('f4'))) is not
            # np.dtype('f4') Because pickle relies on memoization during
            # pickling, it is easy to
            # produce different hashes for seemingly identical objects, such as
            # ``[np.dtype('f4'), np.dtype('f4')]``
            # and ``[np.dtype('f4'), pickle.loads(pickle.dumps('f4'))]``.
            # To prevent memoization from interfering with hashing, we isolate
            # the serialization (and thus the pickle memoization) of each dtype
            # using each time a different ``pickle.dumps`` call unrelated to
            # the current Hasher instance.
            self._hash.update("_HASHED_DTYPE".encode("utf-8"))
            self._hash.update(pickle.dumps(obj))
            return
        Hasher.save(self, obj)


def hash(obj, hash_name="md5", coerce_mmap=False):
    """Quick calculation of a hash to identify uniquely Python objects
    containing numpy arrays.


    Parameters
    -----------
    hash_name: 'md5' or 'sha1'
        Hashing algorithm used. sha1 is supposedly safer, but md5 is
        faster.
    coerce_mmap: boolean
        Make no difference between np.memmap and np.ndarray
    """
    valid_hash_names = ("md5", "sha1")
    if hash_name not in valid_hash_names:
        raise ValueError(
            "Valid options for 'hash_name' are {}. "
            "Got hash_name={!r} instead.".format(valid_hash_names, hash_name)
        )
    if "pandas" in sys.modules:
        hasher = xxPandasHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    elif "numpy" in sys.modules:
        hasher = xxNumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = xxHasher(hash_name=hash_name)
    return hasher.hash(obj)


class DynamicMemorizedFunc(MemorizedFunc):
    def __init__(
        self,
        func,
        location,
        backend="local",
        ignore=None,
        mmap_mode=None,
        compress=False,
        verbose=1,
        timestamp=None,
        min_time_to_cache=10,
    ):
        super().__init__(
            func=func,
            location=location,
            backend=backend,
            ignore=ignore,
            mmap_mode=mmap_mode,
            compress=compress,
            verbose=verbose,
            timestamp=timestamp,
        )
        self.min_time_to_cache = min_time_to_cache

    def _get_argument_hash(self, *args, **kwargs):
        return hash(
            filter_args(self.func, self.ignore, args, kwargs),
            coerce_mmap=(self.mmap_mode is not None),
        )

    def call(self, *args, **kwargs):
        """Force the execution of the function with the given arguments and
        persist the output values.
        """
        start_time = time.time()
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        if self._verbose > 0:
            print(format_call(self.func, args, kwargs))
        func_start_time = time.time()
        output = self.func(*args, **kwargs)
        func_duration = time.time() - func_start_time
        if func_duration >= self.min_time_to_cache:
            self.store_backend.dump_item(
                [func_id, args_id], output, verbose=self._verbose
            )

            duration = time.time() - start_time
            metadata = self._persist_input(duration, args, kwargs)

            if self._verbose > 0:
                _, name = get_func_name(self.func)
                msg = "%s - %s" % (name, format_time(duration))
                print(max(0, (80 - len(msg))) * "_" + msg)
        else:
            if self._verbose > 0:
                _, name = get_func_name(self.func)
                msg = "%s - not caching as it took %s" % (
                    name,
                    format_time(func_duration),
                )
                print(max(0, (80 - len(msg))) * "_" + msg)
            metadata = None
        return output, metadata

    def _cached_call(self, args, kwargs, shelving=False):
        """Call wrapped function and cache result, or read cache if available.

        This function returns the wrapped function output and some metadata.

        Arguments:
        ----------

        args, kwargs: list and dict
            input arguments for wrapped function

        shelving: bool
            True when called via the call_and_shelve function.


        Returns
        -------
        output: value or tuple or None
            Output of the wrapped function.
            If shelving is True and the call has been already cached,
            output is None.

        argument_hash: string
            Hash of function arguments.

        metadata: dict
            Some metadata about wrapped function call (see _persist_input()).
        """
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        metadata = None
        msg = None

        # Wether or not the memorized function must be called
        must_call = False

        # FIXME: The statements below should be try/excepted
        # Compare the function code with the previous to see if the
        # function code has changed
        if not (
            self._check_previous_func_code(stacklevel=4)
            and self.store_backend.contains_item([func_id, args_id])
        ):
            if self._verbose > 10:
                _, name = get_func_name(self.func)
                self.warn(
                    "Computing func {0}, argument hash {1} "
                    "in location {2}".format(
                        name,
                        args_id,
                        self.store_backend.get_cached_func_info([func_id])["location"],
                    )
                )
            must_call = True
        else:
            try:
                t0 = time.time()
                if self._verbose:
                    msg = _format_load_msg(
                        func_id, args_id, timestamp=self.timestamp, metadata=metadata
                    )

                if not shelving:
                    # When shelving, we do not need to load the output
                    out = self.store_backend.load_item(
                        [func_id, args_id], msg=msg, verbose=self._verbose
                    )
                else:
                    out = None

                if self._verbose > 4:
                    t = time.time() - t0
                    _, name = get_func_name(self.func)
                    msg = "%s cache loaded - %s" % (name, format_time(t))
                    print(max(0, (80 - len(msg))) * "_" + msg)
            except Exception:
                # XXX: Should use an exception logger
                _, signature = format_signature(self.func, *args, **kwargs)
                self.warn(
                    "Exception while loading results for "
                    "{}\n {}".format(signature, traceback.format_exc())
                )

                must_call = True

        if must_call:
            out, metadata = self.call(*args, **kwargs)
            if self.mmap_mode is not None and metadata is not None:
                # Memmap the output at the first call to be consistent with
                # later calls
                if self._verbose:
                    msg = _format_load_msg(
                        func_id, args_id, timestamp=self.timestamp, metadata=metadata
                    )
                out = self.store_backend.load_item(
                    [func_id, args_id], msg=msg, verbose=self._verbose
                )

        return (out, args_id, metadata)

    def call_and_shelve(self, *args, **kwargs):
        """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. location=None in Memory).
        """
        out, args_id, metadata = self._cached_call(args, kwargs, shelving=True)
        if metadata is None:
            return NotMemorizedResult(out)
        return MemorizedResult(
            self.store_backend,
            self.func,
            args_id,
            metadata=metadata,
            verbose=self._verbose - 1,
            timestamp=self.timestamp,
        )


class DynamicMemory(Memory):
    def __init__(
        self,
        location=None,
        backend="local",
        cachedir=None,
        mmap_mode=None,
        compress=False,
        verbose=0,
        bytes_limit=None,
        backend_options=None,
        min_time_to_cache=8,
    ):
        super().__init__(
            location=location,
            backend=backend,
            cachedir=cachedir,
            mmap_mode=mmap_mode,
            compress=compress,
            verbose=verbose,
            bytes_limit=bytes_limit,
            backend_options=backend_options,
        )
        self.min_time_to_cache = min_time_to_cache

    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False):
        """Decorates the given function func to only compute its return
        value for input arguments not cached on disk.

        Parameters
        ----------
        func: callable, optional
            The function to be decorated
        ignore: list of strings
            A list of arguments name to ignore in the hashing
        verbose: integer, optional
            The verbosity mode of the function. By default that
            of the memory object is used.
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments. By default that of the memory object is used.

        Returns
        -------
        decorated_func: MemorizedFunc object
            The returned object is a MemorizedFunc object, that is
            callable (behaves like a function), but offers extra
            methods for cache lookup and management. See the
            documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(
                self.cache, ignore=ignore, verbose=verbose, mmap_mode=mmap_mode
            )
        if self.store_backend is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, MemorizedFunc):
            func = func.func
        return DynamicMemorizedFunc(
            func,
            location=self.store_backend,
            backend=self.backend,
            ignore=ignore,
            mmap_mode=mmap_mode,
            compress=self.compress,
            verbose=verbose,
            timestamp=self.timestamp,
            min_time_to_cache=self.min_time_to_cache,
        )


def dynamic_memory_factory(cache, **kwargs):
    memory = tempfile.gettempdir() if cache is True else cache
    memory = memory if not memory == os.getcwd() else ".."
    return DynamicMemory(memory, **kwargs)