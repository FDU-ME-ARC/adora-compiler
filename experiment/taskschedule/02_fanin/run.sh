#!/bin/bash
# 02_fanin/run.sh — two parallel Loads fan-in to Kernel
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"
CGRA_OPT="${ROOT}/build/bin/cgra-opt"
FC="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck"

echo "=== 02_fanin: LoadA,LoadB (parallel) → kernel async[tok0,tok1] → Store ==="
echo "    LoadA and LoadB are independent: can run on different DMA streams"
echo ""

"${CGRA_OPT}" "${DIR}/input.mlir" \
    --adora-schedule-tasks="dump-token-graph=${DIR}/tokens.dot" \
    2>/dev/null > "${DIR}/output_token.mlir"

echo "--- token chain ---"
grep -E "async|!ADORA\.token" "${DIR}/output_token.mlir" | sed 's/^/  /'

echo ""
echo "--- assign-streams (parallel DMA allocation) ---"
"${CGRA_OPT}" "${DIR}/output_token.mlir" --adora-assign-streams 2>/dev/null \
    > "${DIR}/output_streams.mlir" || true
grep "stream" "${DIR}/output_streams.mlir" 2>/dev/null | sed 's/^/  /' || echo "  (skipped)"

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
