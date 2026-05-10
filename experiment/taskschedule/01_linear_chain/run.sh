#!/bin/bash
# 01_linear_chain/run.sh — single Load→Kernel→Store token chain
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"
CGRA_OPT="${ROOT}/build/bin/cgra-opt"
FC="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck"

echo "=== 01_linear_chain: Load → Kernel → Store ==="
echo "    expected: BlockLoad→tok0 → kernel async[tok0]→tokK → BlockStore async[tokK]"
echo ""

# 1. run pass
"${CGRA_OPT}" "${DIR}/input.mlir" \
    --adora-schedule-tasks="emit-token=true dump-token-graph=${DIR}/tokens.dot" \
    2>/dev/null > "${DIR}/output_token.mlir"

# 2. show token chain
echo "--- token chain ---"
grep -E "async|!ADORA\.token" "${DIR}/output_token.mlir" | sed 's/^/  /'

# 3. FileCheck
echo ""
echo "--- FileCheck ---"
"${CGRA_OPT}" "${DIR}/input.mlir" \
    --adora-schedule-tasks="emit-token=true" \
    2>/dev/null | "${FC}" "${DIR}/check.mlir"
echo "  PASSED ✓"

# 4. dot
if command -v dot &>/dev/null && [[ -f "${DIR}/tokens.dot" ]]; then
    dot -Tpng "${DIR}/tokens.dot" -o "${DIR}/tokens.png" 2>/dev/null
    echo ""
    echo "--- tokens.png generated ---"
fi
echo ""
echo "DONE"
