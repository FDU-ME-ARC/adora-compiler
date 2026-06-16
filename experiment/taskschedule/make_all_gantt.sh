#!/bin/bash
# make_all_gantt.sh — generate PE-array + SPAD occupancy Gantt figures for every
# benchmark that has real kernels + DMA (01..10). Benchmarks 11..18 are skipped
# (no kernels / no BlockLoad-Store -> nothing to schedule; see SWEEP_RESULTS).
#
# For each benchmark:
#   1. cgra-opt --llm-pipeline-schedule (dryrule ranker -> round-robin tiles)
#   2. cycle-estimator run.py --viz / --viz-sram
#
# Output: <bench>/_gantt/<bench>_gantt.{pdf,png} + _sram.{pdf,png}
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"               # adora-compiler/
AGENT_ROOT="$(cd "${ROOT}/.." && pwd)"           # aicb-agent/

CGRA_OPT="${ROOT}/build/bin/cgra-opt"
ESTIMATOR="${ROOT}/tools/cycle-estimator/run.py"
RANKER="${AGENT_ROOT}/experiments/llm_pipeline_tuning/task_schedule_ranker.py"
export ADORA_MLIR_CORE="${ADORA_MLIR_CORE:-/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/python_packages/mlir_core}"

# Only these have real kernels + DMA (verified by SWEEP_RESULTS_20260615.md).
BENCHES=(01_linear_chain 02_fanin 03_3mm 04_gemm_tiled 05_loop_carried \
         06_gray 07_tiled_matmul 08_fir 09_gesummv 10_3mm)

FILTER="${1:-}"
ok=0; skip=0; fail=0
echo "=== Generating Gantt figures for task-schedule benchmarks ==="
for b in "${BENCHES[@]}"; do
  [[ -n "${FILTER}" && "${b}" != *"${FILTER}"* ]] && continue
  IN="${DIR}/${b}/input.mlir"
  [[ -f "${IN}" ]] || { echo "  ${b}: no input.mlir, skip"; skip=$((skip+1)); continue; }
  OUTDIR="${DIR}/${b}/_gantt"
  mkdir -p "${OUTDIR}"
  SCHED="${OUTDIR}/${b}_sched.mlir"
  FIG="${OUTDIR}/${b}_gantt.pdf"
  FIGS="${OUTDIR}/${b}_sram.pdf"

  # 1) schedule (tile_set + dep_type)
  if ! "${CGRA_OPT}" "${IN}" \
        --llm-pipeline-schedule \
        --llm-pipeline-schedule-ranker-cmd="python3 ${RANKER} --backend dryrule" \
        --llm-pipeline-schedule-num-tiles=2 --llm-pipeline-schedule-pe-per-tile=16 \
        > "${SCHED}" 2>/dev/null; then
    echo "  ${b}: schedule FAILED"; fail=$((fail+1)); continue
  fi
  ntile=$(grep -c "adora.tile_set" "${SCHED}" || true)

  # 2) render
  if python3 "${ESTIMATOR}" --mlir "${SCHED}" --cgra-opt "${CGRA_OPT}" \
        --num-alus 16 --num-tiles 2 --viz "${FIG}" --viz-sram "${FIGS}" \
        >/dev/null 2>"${OUTDIR}/viz_stderr.txt"; then
    if [[ -f "${FIG}" && -f "${FIGS}" ]]; then
      echo "  ${b}: OK (${ntile} kernels) -> ${OUTDIR}/"
      ok=$((ok+1))
    else
      echo "  ${b}: viz ran but no figure (see ${OUTDIR}/viz_stderr.txt)"
      fail=$((fail+1))
    fi
  else
    echo "  ${b}: viz FAILED (see ${OUTDIR}/viz_stderr.txt)"
    fail=$((fail+1))
  fi
done
echo "============================================================"
echo "  figures: ${ok} ok, ${fail} failed, ${skip} skipped"
echo "  (11..18 intentionally excluded: no kernels/DMA — see SWEEP_RESULTS)"
echo "============================================================"
[[ "${fail}" -eq 0 ]]
