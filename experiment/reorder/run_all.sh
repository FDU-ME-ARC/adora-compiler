#!/bin/bash
# experiment/reorder/run_all.sh — run every reorder example and FileCheck it.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"

# Prefer the standard build dir; fall back to build-reorder if cgra-opt there
# is stale/absent.
CGRA_OPT="${ROOT}/build/bin/cgra-opt"
if [[ ! -x "${CGRA_OPT}" ]]; then
    CGRA_OPT="${ROOT}/build-reorder/bin/cgra-opt"
fi
FC="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck"

echo "cgra-opt : ${CGRA_OPT}"
echo ""

for ex in "${DIR}"/[0-9]*/; do
    name="$(basename "${ex}")"
    echo "=== ${name} ==="
    echo "--- input nest ---"
    grep -E "affine.for|affine.load|affine.store" "${ex}/input.mlir" | sed 's/^/  /'

    "${CGRA_OPT}" "${ex}/input.mlir" --adora-loop-reorder 2>/dev/null > "${ex}/output.mlir"
    echo "--- reordered nest ---"
    grep -E "affine.for|affine.load|affine.store" "${ex}/output.mlir" | sed 's/^/  /'

    echo "--- FileCheck ---"
    "${FC}" "${ex}/check.mlir" < "${ex}/output.mlir" && echo "  PASSED ✓"
    echo ""
done

echo "DONE"
