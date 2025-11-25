"""
Caching utilities for functions that accept JAX arrays.

This module exposes `memoize_jax`, a decorator that adds **hash-based
memoization** to any Python function whose positional or keyword
arguments include JAX ``DeviceArray`` objects.

Background
----------
JAX arrays are mutable and therefore *unhashable*, so they cannot be
passed directly to ``functools.lru_cache``.  `memoize_jax` solves
this by turning every array into a fully hashable key composed of its
raw bytes, shape and dtype.  The wrapped function receives the original
arrays, while cache lookup happens on the hashable representation.
Internally the cache is maintained by ``functools.lru_cache`` with
``maxsize=None`` (an unbounded cache).
"""

import jax.numpy as np
from functools import lru_cache, wraps


def memoize_jax(func):
    """Memoize functions with JAX array arguments."""

    @lru_cache(maxsize=None)
    def cached_func(*hashable_args, **hashable_kwargs):
        args = [
            (
                np.frombuffer(arg[0], dtype=arg[2]).reshape(arg[1])
                if isinstance(arg, tuple) and len(arg) == 3
                else arg
            )
            for arg in hashable_args
        ]
        kwargs = {
            k: (
                np.frombuffer(v[0], dtype=v[2]).reshape(v[1])
                if isinstance(v, tuple) and len(v) == 3
                else v
            )
            for k, v in hashable_kwargs.items()
        }
        return func(*args, **kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        hashable_args = tuple(
            (
                (arg.tobytes(), arg.shape, arg.dtype)
                if hasattr(arg, "shape") and hasattr(arg, "dtype")
                else arg
            )
            for arg in args
        )
        hashable_kwargs_dict = {
            k: (
                (v.tobytes(), v.shape, v.dtype)
                if hasattr(v, "shape") and hasattr(v, "dtype")
                else v
            )
            for k, v in kwargs.items()
        }
        return cached_func(*hashable_args, **hashable_kwargs_dict)

    return wrapper
