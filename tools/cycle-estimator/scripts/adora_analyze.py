"""Analyze ADORA kernels using the official MLIR Python infrastructure with
the out-of-tree ADORA dialect registered (strong-typed op views).

No regex, no lowering: the original ADORA IR is parsed by upstream mlir.ir
and walked as strong-typed ADORA / affine ops.

Usage:
    PYTHONPATH=<mlir_core>:<build>/lib/CAPI python3 adora_analyze.py kernel.mlir
"""

import sys

from mlir.ir import Context, Module
from adora_mlir import ADORA, register_dialect


def _const_from_affine_map(map_attr_str):
    # e.g. "affine_map<() -> (100)>" -> 100 ; returns None if non-constant.
    import re

    m = re.search(r"->\s*\((-?\d+)\)", map_attr_str)
    return int(m.group(1)) if m else None


def trip_count(for_op):
    attrs = {a.name: str(a.attr) for a in for_op.operation.attributes}
    lb = _const_from_affine_map(attrs.get("lowerBoundMap", ""))
    ub = _const_from_affine_map(attrs.get("upperBoundMap", ""))
    step_s = attrs.get("step", "1 : index").split(":")[0].strip()
    try:
        step = int(step_s)
    except ValueError:
        step = 1
    if lb is None or ub is None or step == 0:
        return None  # symbolic / dynamic bound
    return (ub - lb + step - 1) // step


def analyze(path):
    ctx = Context()
    register_dialect(ctx)
    module = Module.parse(open(path).read(), ctx)

    result = {"kernels": [], "loads": [], "stores": [], "loops": []}

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    view = o.operation.opview
                    name = o.operation.name
                    if isinstance(view, ADORA.KernelOp):
                        kn = {a.name: str(a.attr) for a in o.operation.attributes}
                        result["kernels"].append(kn.get("KernelName", "<unnamed>"))
                    elif isinstance(view, ADORA.DataBlockLoadOp):
                        result["loads"].append(name)
                    elif isinstance(view, ADORA.DataBlockStoreOp):
                        result["stores"].append(name)
                    elif name == "affine.for":
                        result["loops"].append(trip_count(o.operation))
                    walk(o.operation)

    walk(module.operation)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    r = analyze(sys.argv[1])
    print("kernels :", r["kernels"])
    print("loops   :", r["loops"], "(trip counts; None = symbolic)")
    print("loads   :", len(r["loads"]), "ADORA.BlockLoad")
    print("stores  :", len(r["stores"]), "ADORA.BlockStore")
