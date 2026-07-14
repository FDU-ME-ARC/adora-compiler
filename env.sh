#!/usr/bin/env bash
# env.sh -- one-shot loader for the adora-compiler build artifacts.
#
# Usage (MUST be sourced, not executed -- otherwise PATH/vars won't persist
# in your current shell):
#     source env.sh
#   or
#     . env.sh
#
# After loading, the compiled tools are on PATH and you can run, e.g.:
#     cgra-opt  --help                       # MLIR pass driver
#     cgra-mapper  --help                    # CGRA mapper
#     adoracc.py  <kernel.c> ...             # C -> kernel MLIR driver
#     cycle-estimator  --help                # event-level cycle simulator
#     adora-nncompiler.py  ...               # NN compile driver
# --------------------------------------------------------------------------

# Guard: make sure this file is sourced (BASH_SOURCE != $0 means sourced).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "[env.sh] Please load with 'source env.sh'; do not execute directly." >&2
  exit 1
fi

# ---- Repo root (derived from this script's location; source from anywhere) --
ADORA_COMPILER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ADORA_COMPILER

# ---- Build artifacts --------------------------------------------------------
# build/bin holds every compiled tool plus the python entry scripts:
#   cgra-opt  cgra-mapper  tensor-opt  pypack
#   adoracc.py  adora-nncompiler.py  cycle-estimator
export ADORA_BUILD_BIN="${ADORA_COMPILER}/build/bin"
export ADORACC="${ADORA_BUILD_BIN}/adoracc.py"
export ESTIMATOR="${ADORA_BUILD_BIN}/cycle-estimator"

# ---- ONNX frontend (adora-onnx-mlir) ----------------------------------------
# If the onnx-mlir frontend is built, put its Debug/bin on PATH.
ADORA_ONNX_BUILD="${ADORA_COMPILER}/frontend/adora-onnx-mlir/build"
ADORA_ONNX_BIN="${ADORA_ONNX_BUILD}/Debug/bin"
if [ -x "${ADORA_ONNX_BIN}/onnx-mlir" ]; then
  export ADORA_ONNX_BIN
  case ":${PATH}:" in
    *":${ADORA_ONNX_BIN}:"*) : ;;
    *) export PATH="${ADORA_ONNX_BIN}:${PATH}" ;;
  esac
  echo "[env.sh] adora-onnx-mlir frontend detected; ${ADORA_ONNX_BIN} added to PATH."
else
  echo "[env.sh] NOTE: adora-onnx-mlir frontend not built at ${ADORA_ONNX_BIN}; skipping." >&2
fi

# ---- External deps (Polygeist frontend cgeist / MLIR python bindings) -------
# cgeist (Polygeist); adoracc finds it on PATH via shutil.which.
export CGEIST_DIR="${ADORA_COMPILER}/frontend/Polygeist/build/bin"
# estimator reads ADORA_MLIR_CORE to inject the `import mlir` bindings.
export ADORA_LLVM_BUILD="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build"
export ADORA_MLIR_CORE="${ADORA_LLVM_BUILD}/python_packages/mlir_core"
# LLVM tool bin (llvm-symbolizer etc.) -- lets crash stack dumps show symbol names.
export ADORA_LLVM_BIN="${ADORA_LLVM_BUILD}/bin"
if [ -x "${ADORA_LLVM_BIN}/llvm-symbolizer" ]; then
  export LLVM_SYMBOLIZER_PATH="${ADORA_LLVM_BIN}/llvm-symbolizer"
fi

# ---- Hardware spec ----------------------------------------------------------
export VITRA_SPEC="${ADORA_COMPILER}/test/spec/cgra_bf16/vitra_spec.json"

# ---- General op-name table --------------------------------------------------
# NB: the binary reads getenv("GeneralOpNameFile") (see include/ADORA/Misc/DFG.h);
# the "GENERAL_OP_NAME_ENV" spelling only appears in the diagnostic message text.
export GeneralOpNameFile="${ADORA_COMPILER}/lib/DFG/Documents/GeneralOpName.txt"
export GENERAL_OP_NAME_ENV="${GeneralOpNameFile}"

# ---- PATH: build/bin first, then cgeist, keep existing PATH -----------------
# Use a case guard so re-sourcing does not append the same entry repeatedly.
case ":${PATH}:" in
  *":${ADORA_BUILD_BIN}:"*) : ;;
  *) export PATH="${ADORA_BUILD_BIN}:${CGEIST_DIR}:${PATH}" ;;
esac

# LLVM build bin (llvm-symbolizer, mlir tools) -- append if present.
if [ -d "${ADORA_LLVM_BIN}" ]; then
  case ":${PATH}:" in
    *":${ADORA_LLVM_BIN}:"*) : ;;
    *) export PATH="${PATH}:${ADORA_LLVM_BIN}" ;;
  esac
fi

# ---- PYTHONPATH: let the estimator import mlir bindings from any cwd --------
case ":${PYTHONPATH:-}:" in
  *":${ADORA_MLIR_CORE}:"*) : ;;
  *) export PYTHONPATH="${ADORA_MLIR_CORE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# ---- Self-check: report any missing path immediately -----------------------
_adora_env_check() {
  local miss=0 p
  for p in \
    "${ADORA_BUILD_BIN}/cgra-opt" \
    "${ADORA_BUILD_BIN}/cgra-mapper" \
    "${ADORACC}" \
    "${ESTIMATOR}" \
    "${CGEIST_DIR}/cgeist" \
    "${ADORA_MLIR_CORE}" \
    "${VITRA_SPEC}" \
    "${GeneralOpNameFile}" ; do
    if [ ! -e "${p}" ]; then echo "  [missing] ${p}" >&2; miss=1; fi
  done
  return ${miss}
}

if _adora_env_check; then
  echo "[env.sh] adora-compiler environment loaded  (ADORA_COMPILER=${ADORA_COMPILER})"
  echo "         cgra-opt / cgra-mapper / adoracc.py / cycle-estimator / cgeist on PATH."
else
  echo "[env.sh] WARNING: the paths listed above are missing; some steps may fail (other vars are still set)." >&2
fi
