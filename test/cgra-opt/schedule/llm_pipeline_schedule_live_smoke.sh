#!/usr/bin/env bash
# Live smoke test for LLMPipelineSchedule: actually calls the Ark LLM via
# task_schedule_ranker.py (--backend openai) and checks that a legal
# hw_dep_type is written into the IR.
#
# This is NOT a lit test: it requires network + a real API key and the LLM
# response is non-deterministic, so we only assert that hw_dep_type appears
# and is a member of the legal set.  The deterministic property check lives in
# llm_pipeline_schedule_dryrun.mlir (run under lit).
#
# Required env:
#   OPENAI_API_KEY   Ark API key
# Optional env (have sane Ark defaults below if unset):
#   PTL_BASE_URL     default https://ark.cn-beijing.volces.com/api/v3
#   PTL_MODEL        default doubao-seed-2-0-code-preview-260215
#   PTL_TIMEOUT      default 120 (seconds; Ark cold start can be slow)
#   CGRA_OPT         path to cgra-opt binary (default: build/bin/cgra-opt)
#
# Usage:
#   OPENAI_API_KEY=sk-... ./llm_pipeline_schedule_live_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo root = .../adora-compiler  (test/cgra-opt/schedule -> up 3)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
AGENT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

CGRA_OPT="${CGRA_OPT:-${REPO_ROOT}/build/bin/cgra-opt}"
RANKER="${AGENT_ROOT}/experiments/llm_pipeline_tuning/task_schedule_ranker.py"

export PTL_BASE_URL="${PTL_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export PTL_TIMEOUT="${PTL_TIMEOUT:-120}"

# Ark is reachable from this intranet; the corporate proxy 403s api.openai.com,
# so route Ark traffic around the proxy.
export no_proxy="${no_proxy:-},.volces.com"
export NO_PROXY="${NO_PROXY:-},.volces.com"

# Auto-load Ark credentials from the notes file when not provided via env.
ARK_NOTES="${AGENT_ROOT}/Agent-Compiler-notes/30_llm_and_api/APIUsingnotes/useAPI_volces_ark.md"
if [[ -z "${OPENAI_API_KEY:-}" && -f "${ARK_NOTES}" ]]; then
  OPENAI_API_KEY="$(grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "${ARK_NOTES}" | head -1)"
  export OPENAI_API_KEY
  if [[ -z "${PTL_MODEL:-}" ]]; then
    PTL_MODEL="$(grep -oE 'ep-[0-9]+-[a-z0-9]+' "${ARK_NOTES}" | head -1)"
  fi
fi
export PTL_MODEL="${PTL_MODEL:-doubao-seed-2-0-code-preview-260215}"

# ---- preconditions ----
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "SKIP: no Ark key (set OPENAI_API_KEY or provide ${ARK_NOTES})" >&2
  exit 77   # automake convention for "skipped"
fi
[[ -x "${CGRA_OPT}" ]] || { echo "FAIL: cgra-opt not found at ${CGRA_OPT}" >&2; exit 1; }
[[ -f "${RANKER}"  ]] || { echo "FAIL: ranker not found at ${RANKER}" >&2; exit 1; }

# ---- test IR: two independent tasks (no RAW dep) -> all dep_types legal ----
IR="$(mktemp /tmp/llm_sched_live.XXXXXX.mlir)"
trap 'rm -f "${IR}"' EXIT
cat > "${IR}" <<'EOF'
module {
  func.func @two_tasks(%arg0: memref<?x25xf32>, %arg1: memref<?x25xf32>) {
    %l0 = "ADORA.BlockLoad"(%arg0)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k0", map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>
    "ADORA.kernel"() ({ "ADORA.terminator"() : () -> () }) {KernelName = "k0"} : () -> ()
    "ADORA.BlockStore"(%l0, %arg0)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "1", KernelName = "k0", map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()
    %l1 = "ADORA.BlockLoad"(%arg1)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "2", KernelName = "k1", map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>
    "ADORA.kernel"() ({ "ADORA.terminator"() : () -> () }) {KernelName = "k1"} : () -> ()
    "ADORA.BlockStore"(%l1, %arg1)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "3", KernelName = "k1", map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()
    return
  }
}
EOF

RANKER_LOG="$(mktemp /tmp/llm_sched_live_log.XXXXXX.ndjson)"
trap 'rm -f "${IR}" "${RANKER_LOG}"' EXIT

echo "== Running cgra-opt with LIVE Ark ranker (model=${PTL_MODEL}) =="
OUT="$("${CGRA_OPT}" "${IR}" \
    --adora-llm-pipeline-schedule \
    --adora-llm-pipeline-schedule-ranker-cmd="python3 ${RANKER} --backend openai --base-url ${PTL_BASE_URL} --model ${PTL_MODEL} --timeout ${PTL_TIMEOUT}" \
    --adora-llm-pipeline-schedule-ranker-timeout=$(( ${PTL_TIMEOUT%.*} * 1000 )) \
    --adora-llm-pipeline-schedule-ranker-log="${RANKER_LOG}" \
    2>/tmp/llm_sched_live.stderr)"

echo "----- IR output -----"
echo "${OUT}" | grep -nE "hw_dep_type|BlockLoad|BlockStore" || true
echo "----- ranker decision log -----"
cat "${RANKER_LOG}" 2>/dev/null || echo "(no log)"
echo "---------------------"

# ---- assertion: a legal hw_dep_type must appear ----
if grep -qE 'hw_dep_type = "(LD_DEP_NONE|LD_DEP_EX_LAST_TASK|LD_DEP_ST_LAST_TASK)"' <<<"${OUT}"; then
  echo "PASS: live LLM produced a legal hw_dep_type"
  exit 0
else
  echo "FAIL: no legal hw_dep_type in output" >&2
  echo "---- stderr ----" >&2
  cat /tmp/llm_sched_live.stderr >&2 || true
  exit 1
fi
