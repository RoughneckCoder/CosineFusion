"""cosinefusion package

This package exposes the Python layer that interfaces with the compiled
binary extension module `cosinefusion.core_backend`.

Usage:
    from cosinefusion import recommend

The compiled extension is available as `cosinefusion.core_backend` and provides
low-level routines such as `cosine_similarity(A, B)` which return a dict with
keys `"similarity_matrix"` and optional `"meta"` information.

The Python layer wraps and exposes higher-level helpers implemented in
`core_demo.py`.
"""

from .core_demo import recommend, main  # re-export high-level helpers
from . import core_backend  # compiled extension module

__all__ = ["recommend", "main", "core_backend"]
