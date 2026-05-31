#!/bin/bash
# 06_mapper_async/run.sh — verify the mapper consumes async tokens end-to-end.
#
# Unlike 01..05 (which exercise the --adora-schedule-tasks pass on skeleton
# kernels), this case runs the full cgra-mapper pipeline on a kernel with a
# real DFG and checks that --enable-async actually changes the emitted code:
#   schedule-tasks -> assign-streams -> lower-async-tokens -> emit
# The schedule-derived token deps must surface as hardware exec deps (C) and
# as depend_type args (pytest, the CGRA-Cocotb-Sim runtime interface).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"
MAPPER="${ROOT}/build/bin/cgra-mapper"
FC="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck"

# Reuse an existing kernel with a real DFG (the taskschedule 01..05 kernels
# have empty bodies and cannot be mapped onto the PE array).
SRC="${ROOT}/test/cgra-mapper/emit/mvt_small_emit.mlir"
ADG="${ROOT}/test/spec/cgra_fp32/cgra_adg_fp32.json"
OP="${ROOT}/test/spec/cgra_fp32/operations_fp32.json"

COMMON=(--adg="${ADG}" --op-file="${OP}" --obj-opt=true --max-iters=1)

echo "=== 06_mapper_async: map mvt kernel with and without async tokens ==="
echo "    src: test/cgra-mapper/emit/mvt_small_emit.mlir"
echo ""

echo "--- baseline (no async): emit C ---"
"${MAPPER}" "${COMMON[@]}" --output-type=c "${SRC}" \
    --output="${DIR}/base.c" >/dev/null 2>&1
echo "  ok ($(wc -l < "${DIR}/base.c") lines)"

echo "--- --enable-async: emit C ---"
"${MAPPER}" --enable-async "${COMMON[@]}" --output-type=c "${SRC}" \
    --output="${DIR}/async.c" >/dev/null 2>&1
echo "  ok ($(wc -l < "${DIR}/async.c") lines)"

echo "--- --enable-async: emit pytest (CGRA-Cocotb-Sim interface) ---"
"${MAPPER}" --enable-async "${COMMON[@]}" --output-type=pytest "${SRC}" \
    --output="${DIR}/async.py" >/dev/null 2>&1
echo "  ok ($(wc -l < "${DIR}/async.py") lines)"

echo ""
echo "--- FileCheck: C async deps ---"
"${FC}" "${DIR}/check_c.txt" < "${DIR}/async.c"
echo "  PASSED ✓"

echo "--- FileCheck: pytest async interface ---"
"${FC}" "${DIR}/check_py.txt" < "${DIR}/async.py"
echo "  PASSED ✓"

echo ""
echo "DONE"
