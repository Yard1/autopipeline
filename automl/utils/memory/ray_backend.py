from typing import Dict
import numpy as np
import cloudpickle
import collections


import ray
from ray.tune.utils import flatten_dict, unflatten_dict

from joblib._store_backends import StoreBackendBase, StoreBackendMixin
from ...search.store import RayStore


class RayStoreBackend(StoreBackendMixin, StoreBackendBase):
    def _item_exists(self, location):
        """Checks if an item location exists in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        location: string
            The location of an item. On a filesystem, this corresponds to the
            absolute path, including the filename, of a file.

        Returns
        -------
        True if the item exists, False otherwise
        """
        return location in self._ref_dict

    def _move_item(self, src, dst):
        """Moves an item from src to dst in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        src: string
            The source location of an item
        dst: string
            The destination location of an item
        """
        pass

    def create_location(self, location):
        """Create object location on store"""
        pass

    def clear_location(self, location):
        """Clears a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory or a filename absolute path
        """
        pass

    def get_items(self):
        """Returns the whole list of items available in the store."""
        pass

    def load_item(self, path, verbose=1, msg=None):
        """Load an item from the store given its path as a list of
           strings."""
        pass

    def dump_item(self, path, item, verbose=1):
        """Dump an item in the store at the path given as a list of
           strings."""
        pass

    def clear_item(self, path):
        """Clear the item at the path, given as a list of strings."""
        pass

    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of
           strings"""
        pass

    def get_item_info(self, path):
        """Return information about item."""
        pass


    def get_metadata(self, path):
        """Return actual metadata of an item."""
        pass

    def store_metadata(self, path, metadata):
        """Store metadata of a computation."""
        pass

    def contains_path(self, path):
        """Check cached function is available in store."""
        pass

    def clear_path(self, path):
        """Clear all items with a common path in the store."""
        pass


    def store_cached_func_code(self, path, func_code=None):
        """Store the code of the cached function."""
        pass

    def get_cached_func_code(self, path):
        """Store the code of the cached function."""
        pass

    def get_cached_func_info(self, path):
        """Return information related to the cached function if it exists."""
        pass

    def clear(self):
        """Clear the whole store content."""
        pass

    def reduce_store_size(self, bytes_limit):
        """Reduce store size to keep it under the given bytes limit."""
        pass

    def configure(self, verbose=1, backend_options=None):
        backend_options = backend_options or {}
        self.compress = False
        self.mmap_mode = None
        self.verbose = verbose
        self._ref_dict = {}
        self.timeout = backend_options.get('timeout', None)

    def __repr__(self):
        """Printable representation of the store location."""
        return f'{self.__class__.__name__}(location="Ray object store")'

