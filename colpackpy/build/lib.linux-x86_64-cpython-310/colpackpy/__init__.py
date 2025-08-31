"""
ColPackPy: Python bindings for ColPack graph coloring library
"""

__version__ = "0.0.1"

# Import the main modules
import os
import sys

# Add the package directory to the library search path
_package_dir = os.path.dirname(__file__)
if hasattr(os, 'add_dll_directory'):  # Windows
    os.add_dll_directory(_package_dir)
else:  # Unix-like systems
    import ctypes.util
    # Try to load the shared library first
    _lib_path = os.path.join(_package_dir, 'libColPack.so.0')
    if os.path.exists(_lib_path):
        try:
            ctypes.CDLL(_lib_path)
        except OSError:
            pass

try:
    from . import coloring
    from . import recovery
except ImportError as e:
    # Handle the case where the C++ extension isn't built yet
    import warnings
    warnings.warn(f"Could not import coloring module: {e}")
    coloring = None

__all__ = ['coloring', 'recovery']
