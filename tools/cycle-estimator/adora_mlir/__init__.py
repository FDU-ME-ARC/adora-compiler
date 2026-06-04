"""Standalone Python package for the out-of-tree ADORA MLIR dialect.

Provides strong-typed ADORA op views on top of the upstream MLIR Python
bindings, plus a `register_dialect(ctx)` helper. No regex, no lowering:
original ADORA IR is parsed and walked as strong-typed ops.

    from adora_mlir import ADORA, register_dialect
    from mlir.ir import Context, Module
    ctx = Context(); register_dialect(ctx)
    m = Module.parse(open("kernel.mlir").read(), ctx)

Requires the upstream MLIR python package (mlir.ir, mlir.dialects) to be
importable (set PYTHONPATH to the mlir_core dir).
"""

import os as _os
import sys as _sys

# Make the bundled _adoraDialectsRegister extension importable.
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

import _adoraDialectsRegister as _reg

from . import _ADORA_ops_gen as ADORA  # strong-typed ADORA op views

# The MLIR C++ runtime resolves strong-typed op views by importing
# `mlir.dialects.<namespace>` (here: mlir.dialects.ADORA). Since we live in
# a standalone package (not inside the upstream tree), alias our module into
# that name so opview resolution finds the ADORA op classes without writing
# any file into the LLVM build tree.
import mlir.dialects as _mlir_dialects  # noqa: F401  (ensure package is set up)

_sys.modules["mlir.dialects.ADORA"] = ADORA
_sys.modules["mlir.dialects._ADORA_ops_gen"] = ADORA


def register_dialect(context, load=True):
    """Register (and optionally load) the ADORA dialect into a context."""
    _reg.register_dialect(context, load)


def register_dialects(registry):
    """Insert the ADORA dialect into a DialectRegistry."""
    _reg.register_dialects(registry)


__all__ = ["ADORA", "register_dialect", "register_dialects"]
