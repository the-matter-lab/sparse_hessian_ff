"""
Shim when the repo root puts this ``colpackpy/`` directory on ``sys.path`` (cwd).

Without a regular ``__init__.py`` here, Python 3 resolves ``import colpackpy``
to this folder as an implicit namespace-only package (PEP 420), never ``src/colpackpy/``,
so ``colpackpy.coloring`` fails even if the extension exists.

Forwarding ``__path__`` lets normal submodule resolution hit ``src/colpackpy``.

The canonical initializer ``src/colpackpy/__init__.py`` runs on first submodule
load (same as a normal ``src``-layout editable install).

See: packaging layout under ``sparse_heffian_ff/colpackpy/`` vs import name ``colpackpy``.
"""

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent / 'src' / 'colpackpy'
__path__ = [str(_PKG_DIR)]
