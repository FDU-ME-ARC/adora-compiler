#!/bin/bash
# 05_loop_carried/run.sh — 64x64x64 tiled GEMM with WAR token + PR6 detection
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"
CGRA_OPT="${ROOT}/build/bin/cgra-opt"
FC="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck"

echo "=== 05_loop_carried: 3-level affine.for, intra-iter WAR token ==="
echo "    C_tile: BlockLoad(C)→%war_tok → kernel → BlockStore async[%war_tok]"
echo "    PR6: loop-carried Store(tk=N)→Load(tk=N+1) detected but not yet wired"
echo ""

"${CGRA_OPT}" "${DIR}/input.mlir" \
    --adora-schedule-tasks="dump-token-graph=${DIR}/tokens.dot" \
    2>"${DIR}/stderr.txt" > "${DIR}/output_token.mlir"

echo "--- token chain ---"
grep -E "async|!ADORA\.token" "${DIR}/output_token.mlir" | sed 's/^/  /'

echo ""
echo "--- PR6 loop-carried detection ---"
if grep -q "PR6-TODO" "${DIR}/stderr.txt" 2>/dev/null; then
    echo "  ✓ Loop-carried dep detected (implementation pending PR6):"
    grep "PR6-TODO\|Store:\|Load:" "${DIR}/stderr.txt" 2>/dev/null | head -4 | sed 's/^/    /'
else
    echo "  (no loop-carried dep detected)"
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
