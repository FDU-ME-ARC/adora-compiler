#!/usr/bin/env bash
# Regenerate _ADORA_ops_gen.py from the ADORA tablegen sources.
# Run this whenever the ADORA op .td files change.
#
# The ADORA ops live in TWO non-including .td files, so we generate each and
# merge, dropping the duplicate _Dialect class from the second.
set -euo pipefail

REPO=/data00/home/loujiahang/adora/adora-compiler-cycleestimator
LLVM=/data00/home/loujiahang/CGRVOPT/llvm-project-onnx
TBLGEN=$LLVM/build/bin/mlir-tblgen
MLIR_INC=$LLVM/mlir/include
LLVM_INC=$LLVM/llvm/include
OUT="$(dirname "$0")/_ADORA_ops_gen.py"

gen() {
  "$TBLGEN" --gen-python-op-bindings -bind-dialect=ADORA \
    -I "$REPO/include" -I "$MLIR_INC" -I "$LLVM_INC" "$1"
}

TMP_OPS=$(mktemp); TMP_KERNEL=$(mktemp)
gen "$REPO/include/ADORA/Dialect/ADORA/IR/ADORAOps.td" > "$TMP_OPS"
gen "$REPO/include/ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.td" > "$TMP_KERNEL"

python3 - "$TMP_OPS" "$TMP_KERNEL" "$OUT" <<'PY'
import sys
ops, kernel, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = open(ops).read()
b = open(kernel).read()
marker = "@_ods_cext.register_operation(_Dialect)"
idx = b.find(marker)
merged = a.rstrip() + "\n\n\n" + (b[idx:] if idx != -1 else "")
# Standalone package: resolve _ods_common against the upstream package.
merged = merged.replace("from ._ods_common import", "from mlir.dialects._ods_common import")
open(out, "w").write(merged)
import re
print("regenerated", out, "with ops:", re.findall(r"class (\w+Op)\(", merged))
PY

rm -f "$TMP_OPS" "$TMP_KERNEL"
