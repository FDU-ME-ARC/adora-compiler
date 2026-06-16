#!/bin/bash
# make_complex_gantt.sh — full C-to-figure pipeline for the complex/ kernels
# (attn/ffn/fft/sobel/viterbi), i.e. the benchmarks 14..18 that ship as C with
# #pragma scop rather than as already-lowered MLIR.
#
# Per kernel:
#   1. adoracc.py  : C (#pragma scop) -> optimized kernel MLIR (kernels + DMA)
#   2. cgra-opt    : --llm-pipeline-schedule (dryrule ranker) -> tile_set/dep_type
#   3. estimator   : --viz / --viz-sram -> Gantt (PE) + SPAD occupancy figures
#
# Output: complex/<k>/_gantt/<k>_gantt.{pdf,png} + _sram.{pdf,png}
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../taskschedule/complex
TS="$(cd "${DIR}/.." && pwd)"                          # .../taskschedule
ROOT="$(cd "${TS}/../.." && pwd)"                      # adora-compiler/
AGENT_ROOT="$(cd "${ROOT}/.." && pwd)"                 # aicb-agent/

CGRA_OPT="${ROOT}/build/bin/cgra-opt"
CGEIST_DIR="/data00/home/loujiahang/adora/adora-compiler/frontend/Polygeist/build/bin"
ADORACC="${ROOT}/tools/adoracc/adoracc.py"
ESTIMATOR="${ROOT}/tools/cycle-estimator/run.py"
RANKER="${AGENT_ROOT}/experiments/llm_pipeline_tuning/task_schedule_ranker.py"

# adoracc needs cgra-opt + cgeist on PATH; estimator needs the mlir bindings.
export PATH="${ROOT}/build/bin:${CGEIST_DIR}:${PATH}"
export ADORA_MLIR_CORE="${ADORA_MLIR_CORE:-/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/python_packages/mlir_core}"

KERNELS=(attn ffn fft sobel viterbi atax jacobi1d cholesky)
FILTER="${1:-}"
ok=0; fail=0
echo "=== C -> Gantt for complex/ kernels (14..18) ==="
for k in "${KERNELS[@]}"; do
  [[ -n "${FILTER}" && "${k}" != *"${FILTER}"* ]] && continue
  C="${DIR}/${k}/${k}.c"
  [[ -f "${C}" ]] || { echo "  ${k}: no ${k}.c, skip"; continue; }
  OUT="${DIR}/${k}/_gantt"
  mkdir -p "${OUT}"
  MLIR="${OUT}/${k}_opt.mlir"
  SCHED="${OUT}/${k}_sched.mlir"
  FIG="${OUT}/${k}_gantt.pdf"
  FIGS="${OUT}/${k}_sram.pdf"

  # 1) C -> kernel MLIR (adoracc runs cgeist + cgra-opt pipeline).
  # NOTE: adoracc's final --adora-kernel-dfg-gen step can abort (exit -6) on some
  # kernels (e.g. cholesky's triangular loops; see SWEEP_RESULTS B4), but the
  # optimized kernel MLIR is written to -o BEFORE that step. We only need the
  # MLIR (the estimator regenerates its own DFG), so we ignore adoracc's exit
  # code and instead check that the MLIR with kernels was produced.
  python3 "${ADORACC}" "${C}" --work-dir "${OUT}/_cc" -o "${MLIR}" \
        >"${OUT}/adoracc.log" 2>&1 || true
  if [[ ! -f "${MLIR}" ]]; then
    echo "  ${k}: adoracc produced no MLIR (see ${OUT}/adoracc.log)"; fail=$((fail+1)); continue
  fi
  nk=$(grep -c "ADORA.kernel" "${MLIR}" || true)
  if [[ "${nk}" -eq 0 ]]; then
    echo "  ${k}: 0 kernels after adoracc (see ${OUT}/adoracc.log)"; fail=$((fail+1)); continue
  fi

  # 2) schedule -> tile_set / dep_type
  if ! "${CGRA_OPT}" "${MLIR}" \
        --llm-pipeline-schedule \
        --llm-pipeline-schedule-ranker-cmd="python3 ${RANKER} --backend dryrule" \
        --llm-pipeline-schedule-num-tiles=2 --llm-pipeline-schedule-pe-per-tile=16 \
        > "${SCHED}" 2>/dev/null; then
    echo "  ${k}: schedule FAILED"; fail=$((fail+1)); continue
  fi
  nt=$(grep -c "adora.tile_set" "${SCHED}" || true)

  # 3) render Gantt + SPAD
  if python3 "${ESTIMATOR}" --mlir "${SCHED}" --cgra-opt "${CGRA_OPT}" \
        --num-alus 16 --num-tiles 2 --viz "${FIG}" --viz-sram "${FIGS}" \
        >/dev/null 2>"${OUT}/viz_stderr.txt" && [[ -f "${FIG}" && -f "${FIGS}" ]]; then
    echo "  ${k}: OK (${nk} kernels, ${nt} tile_set) -> ${OUT}/"
    ok=$((ok+1))
  else
    echo "  ${k}: viz FAILED (see ${OUT}/viz_stderr.txt)"; fail=$((fail+1))
  fi
done
echo "============================================================"
echo "  complex figures: ${ok} ok, ${fail} failed"
echo "============================================================"
[[ "${fail}" -eq 0 ]]
