import itertools
from typing import Optional
import sys
import io

import numpy as np

import pickle
from joblib.hashing import Hasher, NumpyHasher

from pandas.util import hash_pandas_object, hash_array
from pandas.core.util.hashing import _default_hash_key, hash_tuples, _combine_hash_arrays
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndexClass,
    ABCMultiIndex,
    ABCSeries,
)

from xxhash import xxh3_128

Pickler = pickle._Pickler


def fast_hash_pandas_object(
    obj,
    index: bool = True,
    encoding: str = "utf8",
    hash_key: Optional[str] = _default_hash_key,
    categorize: bool = True,
):
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    index : bool, default True
        Include the index in the hash (if Series/DataFrame).
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    Series of uint64, same length as the object
    """
    if hash_key is None:
        hash_key = _default_hash_key

    if isinstance(obj, ABCMultiIndex):
        return hash_tuples(obj, encoding, hash_key)

    elif isinstance(obj, ABCIndexClass):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )

    elif isinstance(obj, ABCSeries):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )
        if index:
            index_iter = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values
                for _ in [None]
            )
            arrays = itertools.chain([h], index_iter)
            h = _combine_hash_arrays(arrays, 2)

    elif isinstance(obj, ABCDataFrame):
        hashes = (hash_array(series._values) for _, series in obj.items())
        num_items = len(obj.columns)
        if index:
            index_hash_generator = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values
                for _ in [None]
            )
            num_items += 1

            # keep `hashes` specifically a generator to keep mypy happy
            _hashes = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)
        h = _combine_hash_arrays(hashes, num_items)

    else:
        raise TypeError(f"Unexpected type for hashing {type(obj)}")
    return h


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
            hashed_obj = fast_hash_pandas_object(obj)
            return (obj.__class__, hashed_obj)
        except TypeError:
            return obj

    def hash(self, obj, return_digest=True):
        try:
            if isinstance(obj, dict):
                n_obj = (obj.__class__, {k: self._hash_pandas(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                n_obj = (obj.__class__, [self._hash_pandas(v) for v in obj])
            elif isinstance(obj, tuple):
                n_obj = (obj.__class__, tuple([self._hash_pandas(v) for v in obj]))
            else:
                n_obj = self._hash_pandas(obj)
            obj = n_obj
        except:
            pass
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
            self._hash.update(self._getbuffer(obj_c_contiguous.view(self.np.uint16)))

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

