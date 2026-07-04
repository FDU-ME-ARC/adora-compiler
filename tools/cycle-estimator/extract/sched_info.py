"""Extract per-kernel SCHEDULING decisions from an ADORA kernel MLIR using the
upstream MLIR Python bindings (strong-typed), with no regex on the IR body.

These attributes are written by the `adora-llm-pipeline-schedule` pass:
  - `adora.tile_set` (DenseI64ArrayAttr on the KernelOp): which CGRA tile(s)
    the kernel is assigned to.  Printed as `array<i64: 0, 1>`.
  - `hw_dep_type`     (StringAttr, on the kernel's BlockLoad/BlockStore ops that
    carry the same KernelName): e.g. LD_DEP_NONE / LD_DEP_ST_LAST_TASK -- decides
    whether the kernel can overlap with the previous task.

Sibling of `loop_info.py`; reuses the same `adora_mlir` dialect registration and
`_strip_module_attrs` helper (import sets up sys.path for mlir_core + adora_mlir).
The input MLIR must already be scheduled; otherwise tile_set is absent and we
fall back to tile [0] / dep NONE.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

# Importing loop_info runs its module-level sys.path setup (mlir_core +
# adora_mlir) and gives us the brace-matched module-attr stripper.
from .loop_info import _strip_module_attrs  # noqa: E402

_DEP_NONE = "LD_DEP_NONE"


@dataclass
class KernelSched:
    name: str
    tiles: list[int] = field(default_factory=lambda: [0])
    dep_type: str = _DEP_NONE                       # coarse dependency class
    dep_on: list[str] = field(default_factory=list)  # kernel names it waits on

    @property
    def overlaps_prev(self) -> bool:
        """True if this kernel may overlap with prior tasks (no serial barrier).

        Judged SOLELY by dep_type: LD_DEP_NONE means no RAW barrier, so it may
        overlap. Must NOT also test `dep_on` here -- dep_on is DERIVED from this
        flag later (serial-chain backfill), so including it created a circular
        definition (`not dep_on` was always True before backfill, making
        overlaps_prev always True and the dep_on backfill never fire)."""
        return self.dep_type == _DEP_NONE


def _parse_i64_array(attr_str: str) -> list[int]:
    """`array<i64: 0, 1>` -> [0, 1]. Tolerant of spaces / empty."""
    lb = attr_str.find(":")
    rb = attr_str.rfind(">")
    if lb < 0 or rb < 0 or rb <= lb:
        return []
    body = attr_str[lb + 1:rb].strip()
    if not body:
        return []
    out: list[int] = []
    for tok in body.split(","):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            out.append(int(tok))
    return out


def _kernel_name(co) -> str:
    for a in co.attributes:
        if a.name in ("KernelName", "kernel_name", "sym_name"):
            return str(a.attr).strip('"')
    return ""


def _attrs(co) -> dict:
    return {a.name: str(a.attr) for a in co.attributes}


def extract_kernel_sched(mlir_path: str) -> list[KernelSched]:
    """Walk the module; for each ADORA.kernel read adora.tile_set; for any op
    carrying the same KernelName read hw_dep_type. One KernelSched per kernel,
    in source order."""
    import mlir.ir as ir
    import adora_mlir

    with open(mlir_path) as f:
        txt = _strip_module_attrs(f.read())
    ctx = ir.Context()
    adora_mlir.register_dialect(ctx)
    module = ir.Module.parse(txt, ctx)

    scheds: dict[str, KernelSched] = {}
    order: list[str] = []
    dep_by_kernel: dict[str, str] = {}

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    co = child.operation
                    name = co.name
                    attrs = _attrs(co)
                    kname = _kernel_name(co)

                    if name == "ADORA.kernel":
                        ks = scheds.get(kname)
                        if ks is None:
                            ks = KernelSched(name=kname or f"kernel{len(order)}")
                            scheds[kname] = ks
                            order.append(kname)
                        if "adora.tile_set" in attrs:
                            t = _parse_i64_array(attrs["adora.tile_set"])
                            if t:
                                ks.tiles = t

                    # hw_dep_type usually rides on the kernel's IO ops.
                    if "hw_dep_type" in attrs and kname:
                        dep = attrs["hw_dep_type"].strip('"')
                        prev = dep_by_kernel.get(kname)
                        # strongest (non-NONE) dep wins for the kernel
                        if prev is None or (prev == _DEP_NONE and dep != _DEP_NONE):
                            dep_by_kernel[kname] = dep

                    walk(co)

    walk(module.operation)

    for kname, dep in dep_by_kernel.items():
        if kname in scheds:
            scheds[kname].dep_type = dep

    # Coarse serial-chain model: a kernel with a non-NONE dep waits for the
    # immediately preceding kernel in source order.
    prev_name = None
    for kname in order:
        ks = scheds[kname]
        if not ks.overlaps_prev and prev_name is not None:
            ks.dep_on = [prev_name]
        prev_name = kname

    return [scheds[k] for k in order]


def parse_kernel_sched(mlir_path: str) -> dict[str, KernelSched]:
    return {k.name: k for k in extract_kernel_sched(mlir_path)}


if __name__ == "__main__":
    for k in extract_kernel_sched(sys.argv[1]):
        print(f"{k.name}: tiles={k.tiles} dep_type={k.dep_type} "
              f"dep_on={k.dep_on} overlaps_prev={k.overlaps_prev}")
