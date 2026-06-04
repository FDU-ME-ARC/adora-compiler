"""Extract loop trip-counts and memref element widths from an ADORA kernel MLIR
using the upstream MLIR Python bindings (strong-typed), with no regex.

The ADORA out-of-tree dialect is registered via the bundled `adora_mlir`
package; affine.for trip counts are read from the loop's lower/upper bound
affine maps, and element width from the memref element type.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# The upstream MLIR python package (mlir.ir, mlir.dialects) ships with the LLVM
# build. Allow overriding via env, else fall back to the known build location.
_MLIR_CORE = os.environ.get(
    "ADORA_MLIR_CORE",
    "/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/python_packages/mlir_core",
)
if _MLIR_CORE and _MLIR_CORE not in sys.path:
    sys.path.insert(0, _MLIR_CORE)

# Make the sibling adora_mlir package importable regardless of CWD.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


@dataclass
class KernelLoops:
    name: str
    trip_counts: list[int]   # outermost first
    elt_bytes: int

    @property
    def inner_trip(self) -> int:
        return self.trip_counts[-1] if self.trip_counts else 1

    @property
    def outer_trip(self) -> int:
        prod = 1
        for t in self.trip_counts[:-1]:
            prod *= t
        return prod

    @property
    def total_trip(self) -> int:
        prod = 1
        for t in self.trip_counts:
            prod *= t
        return prod


def _strip_module_attrs(text: str) -> str:
    """Drop a leading `module attributes { ... } {` dict that some front-ends
    (e.g. Polygeist's dlti.dl_spec) emit but the MLIR Python parser rejects.
    Replaces it with a plain `module {`. Brace-matched, no regex."""
    i = text.find("module attributes")
    if i < 0:
        return text
    brace = text.find("{", i)
    if brace < 0:
        return text
    depth = 0
    j = brace
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    # j now at the closing brace of the attribute dict; find the module's own '{'
    body = text.find("{", j + 1)
    if body < 0:
        return text
    return text[:i] + "module " + text[body:]


def _const_from_affine_map(map_attr) -> int | None:
    """Return the constant result of a 0-input affine map like
    `affine_map<() -> (100)>`, or None if not a plain constant."""
    try:
        amap = map_attr.value  # AffineMapAttr -> AffineMap
    except AttributeError:
        amap = map_attr
    if amap.n_dims != 0 or amap.n_symbols != 0:
        return None
    if len(amap.results) != 1:
        return None
    expr = amap.results[0]
    # AffineConstantExpr has a `.value`
    try:
        from mlir.ir import AffineConstantExpr
        if isinstance(expr, AffineConstantExpr):
            return expr.value
    except Exception:
        pass
    s = str(expr)
    return int(s) if s.lstrip("-").isdigit() else None


def _for_trip(for_op) -> int | None:
    attrs = {a.name: a.attr for a in for_op.attributes}
    lb = _const_from_affine_map(attrs.get("lowerBoundMap"))
    ub = _const_from_affine_map(attrs.get("upperBoundMap"))
    step_attr = attrs.get("step")
    try:
        step = int(str(step_attr).split(":")[0].strip())
    except Exception:
        step = 1
    if lb is None or ub is None or step <= 0:
        return None
    return max(0, (ub - lb + step - 1) // step)


def _elt_bytes_from_memref(mtype_str: str) -> int:
    # mtype_str like "memref<100xi32>" -> element "i32" -> 4 bytes
    inner = mtype_str.split("x")[-1].rstrip(">")
    for tag, bits in (("i8", 8), ("i16", 16), ("i32", 32), ("i64", 64),
                      ("f16", 16), ("bf16", 16), ("f32", 32), ("f64", 64)):
        if inner == tag:
            return bits // 8
    return 4


def extract_kernel_loops(mlir_path: str) -> list[KernelLoops]:
    import mlir.ir as ir
    import adora_mlir

    with open(mlir_path) as f:
        txt = _strip_module_attrs(f.read())
    ctx = ir.Context()
    adora_mlir.register_dialect(ctx)
    module = ir.Module.parse(txt, ctx)

    out: list[KernelLoops] = []

    def find_kernels(op, parent_name="kernel"):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    co = child.operation
                    name = co.name
                    if name == "ADORA.kernel":
                        kname = parent_name
                        for a in co.attributes:
                            if a.name in ("KernelName", "kernel_name", "sym_name"):
                                kname = str(a.attr).strip('"')
                        trips: list[int] = []
                        elt = 4
                        _collect_loops(co, trips)
                        elt = _first_memref_elt(co, elt)
                        out.append(KernelLoops(kname, trips, elt))
                    find_kernels(co, parent_name)

    def _collect_loops(op, trips):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    co = child.operation
                    if co.name == "affine.for":
                        t = _for_trip(co)
                        if t is not None:
                            trips.append(t)
                        _collect_loops(co, trips)
                    else:
                        _collect_loops(co, trips)

    def _first_memref_elt(op, default):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    co = child.operation
                    if co.name in ("affine.load", "affine.store"):
                        for operand in co.operands:
                            ts = str(operand.type)
                            if ts.startswith("memref<"):
                                return _elt_bytes_from_memref(ts)
                    r = _first_memref_elt(co, None)
                    if r is not None:
                        return r
        return default

    find_kernels(module.operation)
    return out


def parse_kernels(mlir_path: str) -> dict[str, KernelLoops]:
    """Return all kernels in the MLIR keyed by kernel name."""
    return {k.name: k for k in extract_kernel_loops(mlir_path)}


def kernel_for_dot(mlir_path: str, dot_name: str) -> KernelLoops | None:
    """Find the KernelLoops whose name matches a CDFG dot's base name."""
    kernels = extract_kernel_loops(mlir_path)
    by_name = {k.name: k for k in kernels}
    if dot_name in by_name:
        return by_name[dot_name]
    for name, loops in by_name.items():
        if dot_name in name or name in dot_name:
            return loops
    return None


if __name__ == "__main__":
    for k in extract_kernel_loops(sys.argv[1]):
        print(f"{k.name}: trips={k.trip_counts} inner={k.inner_trip} "
              f"outer={k.outer_trip} total={k.total_trip} elt_bytes={k.elt_bytes}")
