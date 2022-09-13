from joblib import Memory
from joblib.memory import (
    MemorizedFunc,
    NotMemorizedFunc,
    MemorizedResult,
    NotMemorizedResult,
    _format_load_msg,
)
from typing import Optional
import functools
import time
import traceback
import tempfile
import os

import numpy as np

from joblib.func_inspect import get_func_name, filter_args
from joblib.func_inspect import format_call
from joblib.func_inspect import format_signature
from joblib.logger import format_time

from .hashing import hash


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
        min_time_to_cache=1,
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
        min_time_to_cache=1,
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


def dynamic_memory_factory(cache, *, run_id=None, **kwargs):
    if isinstance(cache, Memory):
        return cache
    memory = tempfile.gettempdir() if cache is True else cache
    memory = os.path.join(memory, run_id) if run_id else memory
    memory = memory if not memory == os.getcwd() else ".."
    bytes_limit = kwargs.pop("bytes_limit", 10000000000)
    return DynamicMemory(memory, bytes_limit=bytes_limit, **kwargs)
