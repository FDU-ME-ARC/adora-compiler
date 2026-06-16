#!/bin/bash
# 10_tile_overlap/make_figure.sh — generate the PAPER FIGURE (resource-occupancy
# Gantt) for the two-independent-kernels overlap story.
#
# Pipeline:
#   1. cgra-opt --llm-pipeline-schedule (dryrule ranker -> k0=tile0, k1=tile1)
#      => scheduled MLIR carrying adora.tile_set + hw_dep_type
#   2. cycle-estimator run.py --viz => PE-array occupancy Gantt (PDF + PNG)
#
# The two kernels are independent (dep=LD_DEP_NONE) and on different tiles, so
# their compute phases OVERLAP in time — that's the figure's message.
#
# Usage:   ./make_figure.sh
# Output:  _work/two_kernels_gantt.pdf (+ .png)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../../.." && pwd)"            # adora-compiler/
AGENT_ROOT="$(cd "${ROOT}/.." && pwd)"           # aicb-agent/

CGRA_OPT="${ROOT}/build/bin/cgra-opt"
ESTIMATOR="${ROOT}/tools/cycle-estimator/run.py"
RANKER="${AGENT_ROOT}/experiments/llm_pipeline_tuning/task_schedule_ranker.py"
MLIR="${DIR}/two_kernels.mlir"
WORK="${DIR}/_work"
SCHED="${WORK}/two_kernels_sched.mlir"
FIG="${WORK}/two_kernels_gantt.pdf"
FIG_SRAM="${WORK}/two_kernels_sram.pdf"

# MLIR python bindings location (override with ADORA_MLIR_CORE if your build differs)
export ADORA_MLIR_CORE="${ADORA_MLIR_CORE:-/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/python_packages/mlir_core}"

mkdir -p "${WORK}"

echo "=== [1/2] schedule (dryrule -> k0=tile0, k1=tile1) ==="
"${CGRA_OPT}" "${MLIR}" \
  --llm-pipeline-schedule \
  --llm-pipeline-schedule-ranker-cmd="python3 ${RANKER} --backend dryrule" \
  --llm-pipeline-schedule-num-tiles=2 --llm-pipeline-schedule-pe-per-tile=16 \
  2>/dev/null > "${SCHED}"
echo "    tile_set:"
grep -o "adora.tile_set = array<i64:[^>]*>" "${SCHED}" | sed 's/^/      /'

echo "=== [2/2] render PE-array occupancy Gantt + SPAD occupancy ==="
python3 "${ESTIMATOR}" \
  --mlir "${SCHED}" \
  --cgra-opt "${CGRA_OPT}" \
  --num-alus 16 --num-tiles 2 \
  --viz "${FIG}" --viz-sram "${FIG_SRAM}"

echo ""
if [[ -f "${FIG}" && -f "${FIG_SRAM}" ]]; then
  echo "FIGURES WRITTEN:"
  echo "  A (PE occupancy):   ${FIG}   (+ .png)"
  echo "  B (SPAD occupancy): ${FIG_SRAM}   (+ .png)"
  echo "Two independent kernels on different tiles -> compute phases overlap;"
  echo "their SPAD buffers co-reside during the overlap window."
else
  echo "FIGURE FAILED — see output above (matplotlib installed? MLIR scheduled?)"
  exit 1
fi
