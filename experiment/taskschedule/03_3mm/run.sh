#!/bin/bash
# 03_3mm/run.sh — 3-kernel chain with buffer reuse
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"
CGRA_OPT="${ROOT}/build/bin/cgra-opt"
FC="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck"

echo "=== 03_3mm: kernel_0,kernel_1 (parallel) → kernel_2 (fan-in) ==="
echo "    buffer reuse: kernel_2's Load(%arg0) and Load(%arg3) are ELIMINATED"
echo "    on-chip LocalMemAlloc buffers from kernel_0/1 are reused directly"
echo ""

"${CGRA_OPT}" "${DIR}/input.mlir" \
    --adora-schedule-tasks="dump-token-graph=${DIR}/tokens.dot" \
    2>/dev/null > "${DIR}/output_token.mlir"

echo "--- token chain ---"
grep -E "async|!ADORA\.token" "${DIR}/output_token.mlir" | sed 's/^/  /'

echo ""
echo "--- buffer reuse check ---"
if grep -q "ADORA.BlockLoad %arg0" "${DIR}/output_token.mlir"; then
    echo "  WARN: BlockLoad(%arg0) still present"
else
    echo "  ✓ BlockLoad(%arg0) eliminated — kernel_2 uses on-chip buffer directly"
fi
if grep -q "ADORA.BlockLoad %arg3" "${DIR}/output_token.mlir"; then
    echo "  WARN: BlockLoad(%arg3) still present"
else
    echo "  ✓ BlockLoad(%arg3) eliminated — kernel_2 uses on-chip buffer directly"
fi

echo ""
echo "--- FileCheck ---"
"${CGRA_OPT}" "${DIR}/input.mlir" \
    --adora-schedule-tasks \
    2>/dev/null | "${FC}" "${DIR}/check.mlir"
echo "  PASSED ✓"

command -v dot &>/dev/null && [[ -f "${DIR}/tokens.dot" ]] && \
    dot -Tpng "${DIR}/tokens.dot" -o "${DIR}/tokens.png" 2>/dev/null && \
    echo "" && echo "--- tokens.png generated ---"
echo ""
echo "DONE"
