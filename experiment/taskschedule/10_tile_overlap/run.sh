#!/bin/bash
# 10_tile_overlap/run.sh — one-click end-to-end demo + self-check for
#   Stage 1 (TileAssignment writes adora.tile_set) and
#   Stage 2 (mapper constrains each kernel's compute nodes to its tile).
#
# Two INDEPENDENT kernels -> different tiles -> compute placement disjoint.
#
# Usage:   ./run.sh
# Exit 0 = all checks PASS; non-zero = a check FAILED.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"            # adora-compiler/
AGENT_ROOT="$(cd "${ROOT}/.." && pwd)"           # aicb-agent/

CGRA_OPT="${ROOT}/build/bin/cgra-opt"
CGRA_MAPPER="${ROOT}/build/bin/cgra-mapper"
ADG="${ROOT}/test/spec/cgra_fp32/cgra_adg_fp32_2tile.json"     # 2-tile fp32 ADG (fixture)
OPS="${ROOT}/test/spec/cgra_fp32/operations_fp32.json"
RANKER="${AGENT_ROOT}/experiments/llm_pipeline_tuning/task_schedule_ranker.py"
MLIR="${DIR}/two_kernels.mlir"
WORK="${DIR}/_work"

# -------- sanity --------
for f in "${CGRA_OPT}" "${CGRA_MAPPER}" "${ADG}" "${OPS}" "${RANKER}" "${MLIR}"; do
  [[ -e "${f}" ]] || { echo "MISSING: ${f}"; exit 2; }
done
rm -rf "${WORK}"; mkdir -p "${WORK}/trace"
FAIL=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAIL=1; }

echo "============================================================"
echo " 10_tile_overlap: two independent kernels -> different tiles"
echo "============================================================"

# ============================================================
# Check 1 — Stage 1 dialect pass writes adora.tile_set (dry-run, deterministic)
#   dry-run skips the LLM, so each kernel gets the conservative [0..minTiles-1].
# ============================================================
echo ""
echo "--- Check 1: TileAssignment writes adora.tile_set (dry-run) ---"
"${CGRA_OPT}" "${MLIR}" \
  --adora-llm-pipeline-schedule --adora-llm-pipeline-schedule-dry-run \
  --adora-llm-pipeline-schedule-num-tiles=2 --adora-llm-pipeline-schedule-pe-per-tile=16 \
  2>/dev/null > "${WORK}/dryrun.mlir"
N=$(grep -c "adora.tile_set = array<i64:" "${WORK}/dryrun.mlir" || true)
grep -o 'KernelName = "k[01]", adora.tile_set = array<i64:[^>]*>' "${WORK}/dryrun.mlir" | sed 's/^/      /'
[[ "${N}" -eq 2 ]] && pass "both kernels carry adora.tile_set" \
                    || fail "expected 2 tile_set attrs, got ${N}"

# ============================================================
# Check 2 — End-to-end mapping: dryrule ranker round-robins kernels onto
#   DIFFERENT tiles, mapper applies tile constraints, both kernels map,
#   and the COMPUTE nodes land on DISJOINT (tile-correct) GPEs.
# ============================================================
echo ""
echo "--- Check 2: e2e map, compute nodes land on different tiles ---"
( cd "${ROOT}" && "${CGRA_MAPPER}" \
    --adg="${ADG}" --op-file="${OPS}" \
    --output-type=c --obj-opt=true --max-iters=4 \
    --tile=2 --enable-async --enable-llm-schedule \
    --adora-llm-pipeline-schedule-ranker-cmd="python3 ${RANKER} --backend dryrule" \
    --adora-llm-pipeline-schedule-num-tiles=2 --adora-llm-pipeline-schedule-pe-per-tile=16 \
    --emit-agent-trace --agent-trace-root="${WORK}/trace" \
    "${MLIR}" --output="${WORK}/out.c" ) > "${WORK}/map_stdout.log" 2> "${WORK}/map_stderr.log"
MAP_RC=$?
[[ "${MAP_RC}" -eq 0 ]] && pass "cgra-mapper exit 0" || fail "cgra-mapper exit ${MAP_RC}"

# tile_set in the dumped IR must differ between the two kernels
echo "    tile_set written during e2e:"
grep -o "adora.tile_set = array<i64:[^>]*>" "${WORK}/map_stderr.log" | sed 's/^/      /'

# Parse the agent trace to prove compute (FMUL32/FADD32) placement is disjoint.
TRACE=$(ls "${WORK}/trace"/data/mapper_traces/*.jsonl 2>/dev/null | head -1)
if [[ -z "${TRACE}" ]]; then
  fail "no mapper trace produced"
else
  python3 - "${TRACE}" > "${WORK}/trace_check.txt" <<'PYEOF'
import json, sys, collections
tf = sys.argv[1]
ops = {}; final = {}; cur = None; tcons = []
for line in open(tf):
    e = json.loads(line); ev = e.get("event"); d = e.get("data", {})
    if ev == "kernel_start": cur = d.get("kernel")
    if ev == "tile_constraints_applied": tcons.append(d)
    if "dfg_node_id" in d and "operation" in d:
        ops[(d.get("kernel"), d["dfg_node_id"])] = d["operation"]
    if ev in ("map_dfg_node_success", "candidate_accepted"):
        k = d.get("kernel", cur); dn = d.get("dfg_node_id"); an = d.get("adg_node_id")
        if dn is not None and an is not None:
            final[(k, dn)] = an   # keep last = final-ish placement
comp = collections.defaultdict(set)
for (k, dn), an in final.items():
    op = ops.get((k, dn)) or ops.get((None, dn))
    if op in ("FMUL32", "FADD32"):
        comp[k].add(an)
print(f"      tile_constraints_applied events: {len(tcons)}")
for c in tcons:
    print(f"        kernel={c.get('kernel')} tile_set={c.get('tile_set')} "
          f"constrained_nodes={c.get('constrained_nodes')}")
print(f"      k0 compute ADG nodes: {sorted(comp.get('k0', []))}")
print(f"      k1 compute ADG nodes: {sorted(comp.get('k1', []))}")
overlap = comp.get("k0", set()) & comp.get("k1", set())
ok = (len(tcons) == 2) and comp.get("k0") and comp.get("k1") and not overlap
print(f"      compute-placement overlap: {sorted(overlap)}")
print("VERDICT:" + ("PASS" if ok else "FAIL"))
PYEOF
  grep -v "^VERDICT:" "${WORK}/trace_check.txt"
  if grep -q "^VERDICT:PASS" "${WORK}/trace_check.txt"; then
    pass "compute nodes disjoint across tiles (overlap proven)"
  else
    fail "compute placement not disjoint / constraints not applied"
  fi
fi

# ============================================================
echo ""
echo "============================================================"
if [[ "${FAIL}" -eq 0 ]]; then
  echo " ALL CHECKS PASSED ✓   (artifacts in ${WORK})"
  echo "============================================================"
  exit 0
else
  echo " SOME CHECKS FAILED ✗  (see ${WORK}/*.log)"
  echo "============================================================"
  exit 1
fi
