#!/bin/bash
# e2e_pipeline.sh — end-to-end: schedule-tasks → streams → lower-tokens → llvm-runtime
#
# Verifies that !ADORA.token chain produced by adora-schedule-tasks
# correctly flows through all downstream passes to llvm.call runtime ABI.
#
# Usage:
#   ./e2e_pipeline.sh          # run all 4 examples
#   ./e2e_pipeline.sh 02       # run only 02_fanin

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
CGRA_OPT="${ROOT}/build/bin/cgra-opt"

FILTER="${1:-}"

echo "╔══════════════════════════════════════════════╗"
echo "║  End-to-End Async Token Pipeline Verification ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Pipeline: schedule-tasks → assign-streams → lower-async-tokens → to-llvm-async-runtime"
echo ""

ok=0
fail=0

for d in "${DIR}"/*/; do
    name="$(basename "${d}")"
    [[ "${name}" == "__pycache__" ]] && continue
    [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]] && continue
    [[ ! -f "${d}/input.mlir" ]] && continue

    echo -n "▶ ${name}: "

    if ! "${CGRA_OPT}" "${d}/input.mlir" \
        --adora-schedule-tasks \
        --adora-assign-streams \
        --adora-lower-async-tokens \
        --adora-to-llvm-async-runtime \
        > "${d}/e2e_output.mlir" 2>"${d}/e2e_stderr.txt"; then
        echo "FAIL (see e2e_stderr.txt)"
        fail=$((fail + 1))
        continue
    fi

    # Check no unlowered tokens remain.
    # Use grep -q (produces no stdout) to avoid the double-print bug where
    # "grep -c" prints "0" before exiting 1, and "|| echo 0" adds another "0",
    # making $() capture "0\n0" which is != "0" and triggers a false FAIL.
    if grep -q "ADORA\.token" "${d}/e2e_output.mlir" 2>/dev/null; then
        echo "FAIL (unlowered !ADORA.token in output)"
        fail=$((fail + 1))
        continue
    fi

    # Count runtime ABI calls for informational display.
    # Write counts to temp files to avoid the set -euo pipefail + grep-c exit-1 issue.
    _tmp="${d}/.cnt"
    grep -c "adoraEventCreate"  "${d}/e2e_output.mlir" 2>/dev/null > "${_tmp}_create"  || echo 0 > "${_tmp}_create"
    grep -c "adoraEventRecord"  "${d}/e2e_output.mlir" 2>/dev/null > "${_tmp}_record"  || echo 0 > "${_tmp}_record"
    grep -c "adoraEventWait"    "${d}/e2e_output.mlir" 2>/dev/null > "${_tmp}_wait"    || echo 0 > "${_tmp}_wait"
    grep -c "adoraEventDestroy" "${d}/e2e_output.mlir" 2>/dev/null > "${_tmp}_destroy" || echo 0 > "${_tmp}_destroy"
    n_create=$(cat "${_tmp}_create")
    n_record=$(cat "${_tmp}_record")
    n_wait=$(cat "${_tmp}_wait")
    n_destroy=$(cat "${_tmp}_destroy")

    echo "PASS  (create=${n_create} record=${n_record} wait=${n_wait} destroy=${n_destroy})"
    ok=$((ok + 1))
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  Results: ${ok} passed, ${fail} failed"
echo "═══════════════════════════════════════════════"

if [[ "${fail}" -gt 0 ]]; then
    exit 1
fi
