#!/bin/bash
# make_gesummv_gantt.sh — gesummv 专用编译流水（绕开 adoracc 的 --affine-loop-fusion）
#
# 问题：adoracc 在 kernel-extract 阶段跑 --affine-loop-fusion，会把 gesummv 的
#       3 个 stage 循环（Stage1 A*x ∥ Stage2 B*x → Stage3 merge）融成 2 个 kernel，
#       fork-join 结构丢失。
# 方案：手动复现 adoracc 的 cgeist + cgra-opt pass 链，但【去掉 --affine-loop-fusion】，
#       保住 3 个独立 stage → 3 kernel，schedule 才能看到 fork-join RAW 依赖。
#
# Output: gesummv/_gantt/gesummv_{opt,sched}.mlir + gantt/sram 图
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # complex/
TS="$(cd "${DIR}/.." && pwd)"
ROOT="$(cd "${TS}/../.." && pwd)"                              # adora-compiler/
AGENT_ROOT="$(cd "${ROOT}/.." && pwd)"

CGRA_OPT="${ROOT}/build/bin/cgra-opt"
CGEIST="/data00/home/loujiahang/adora/adora-compiler/frontend/Polygeist/build/bin/cgeist"
ESTIMATOR="${ROOT}/tools/cycle-estimator/run.py"
RANKER="${AGENT_ROOT}/experiments/llm_pipeline_tuning/task_schedule_ranker.py"
export ADORA_MLIR_CORE="${ADORA_MLIR_CORE:-/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/python_packages/mlir_core}"

C="${DIR}/gesummv/gesummv.c"
OUT="${DIR}/gesummv/_gantt"
mkdir -p "${OUT}"
FE="${OUT}/gesummv_frontend.mlir"
NORM="${OUT}/gesummv_norm.mlir"
KERN="${OUT}/gesummv_kernel.mlir"
OPT="${OUT}/gesummv_opt.mlir"
SCHED="${OUT}/gesummv_sched.mlir"

echo "=== [1] cgeist: C -> affine MLIR ==="
"${CGEIST}" -O2 --raise-scf-to-affine "${C}" -S -o "${FE}"

echo "=== [2] normalize ==="
"${CGRA_OPT}" --allow-unregistered-dialect --affine-loop-normalize \
  --affine-simplify-structures --normalize-memrefs "${FE}" -o "${NORM}"

echo "=== [3] kernel-extract (NO --affine-loop-fusion -> keep 3 stages) ==="
"${CGRA_OPT}" --canonicalize -reconcile-unrealized-casts \
  --adora-extract-affine-for-to-kernel --arith-expand --memref-expand -cse \
  "${NORM}" -o "${KERN}"

echo "=== [4] kernel-opt ==="
"${CGRA_OPT}" --adora-simplify-affine-loop-levels --canonicalize -cse \
  --adora-simplify-loadstore --adora-math-rewrite \
  "--adora-adjust-kernel-mem-footprint=cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
  "${KERN}" -o "${OPT}"
echo "    kernels=$(grep -c 'ADORA.kernel' "${OPT}")  funcs=$(grep -c 'func.func @' "${OPT}")"

echo "=== [5] llm-pipeline-schedule (dryrule; 改 --backend openai 可走真实 LLM) ==="
"${CGRA_OPT}" "${OPT}" \
  --llm-pipeline-schedule \
  --llm-pipeline-schedule-ranker-cmd="python3 ${RANKER} --backend dryrule" \
  --llm-pipeline-schedule-num-tiles=2 --llm-pipeline-schedule-pe-per-tile=16 \
  > "${SCHED}" 2>/dev/null
echo "    tile_set: $(grep -c 'adora.tile_set' "${SCHED}")  hw_dep_type: $(grep -c 'hw_dep_type' "${SCHED}")"

echo "=== [6] render gantt + sram ==="
python3 "${ESTIMATOR}" --mlir "${SCHED}" --cgra-opt "${CGRA_OPT}" \
  --num-alus 16 --num-tiles 2 \
  --viz "${OUT}/gesummv_gantt.pdf" --viz-sram "${OUT}/gesummv_sram.pdf" \
  2>"${OUT}/viz_stderr.txt" && echo "    figures OK" || { echo "    viz FAILED"; cat "${OUT}/viz_stderr.txt"; }
