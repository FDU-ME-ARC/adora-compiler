#!/usr/bin/env bash
# Build ADORA against LLVM/MLIR. By default, runs build_llvm.sh first if MLIR is missing.
#
# Usage:
#   ./build_tools/build_adora.sh              # LLVM (if needed) + ADORA
#   ./build_tools/build_adora.sh --skip-llvm  # only ADORA; LLVM paths must already be valid
#
# Environment (optional):
#   LLVM_SRC_DIR, LLVM_BUILD_DIR, LLVM_INSTALL_DIR — same meaning as build_llvm.sh
#   BUILD_DIR          ADORA CMake build directory (default: <repo>/build)
#   ADORA_JOBS         Parallel jobs for ADORA build (default: nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SKIP_LLVM=0
for arg in "$@"; do
  if [[ "${arg}" == "--skip-llvm" ]]; then
    SKIP_LLVM=1
  fi
done

mlir_config_path() {
  echo "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake"
}

mlir_installed() {
  [[ -n "${LLVM_INSTALL_DIR:-}" && -f "$(mlir_config_path)" ]]
}

# Align with build_llvm.sh default discovery when LLVM_SRC_DIR is unset.
discover_llvm_src_dir() {
  if [[ -n "${LLVM_SRC_DIR:-}" ]]; then
    [[ -f "${LLVM_SRC_DIR}/llvm/CMakeLists.txt" ]] && return 0
    echo "LLVM_SRC_DIR=${LLVM_SRC_DIR} is not a valid llvm-project root." >&2
    return 1
  fi
  local candidates=(
    "${HOME}/CGRVOPT/llvm-project-onnx"
    "/home/share/onnx-mlir/third_party/llvm-project-onnx"
  )
  for d in "${candidates[@]}"; do
    if [[ -f "${d}/llvm/CMakeLists.txt" ]]; then
      LLVM_SRC_DIR="${d}"
      return 0
    fi
  done
  return 1
}

finalize_llvm_paths() {
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SRC_DIR}/build}"
  LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${LLVM_BUILD_DIR}}"
}

if mlir_installed; then
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_INSTALL_DIR}}"
else
  if [[ "${SKIP_LLVM}" -eq 1 ]]; then
    echo "MLIR not found at $(mlir_config_path) (LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR:-<unset>}) and --skip-llvm was set." >&2
    exit 1
  fi
  if ! discover_llvm_src_dir; then
    echo "Could not locate llvm-project. Set LLVM_SRC_DIR or LLVM_INSTALL_DIR (with an existing MLIR install)." >&2
    exit 1
  fi
  finalize_llvm_paths
  echo "MLIR not found; running ${SCRIPT_DIR}/build_llvm.sh ..."
  export LLVM_SRC_DIR LLVM_BUILD_DIR LLVM_INSTALL_DIR CMAKE_BUILD_TYPE LLVM_JOBS
  "${SCRIPT_DIR}/build_llvm.sh"
  # build_llvm runs in a subprocess; refresh derived paths for this shell
  discover_llvm_src_dir || true
  finalize_llvm_paths
fi

if [[ ! -f "$(mlir_config_path)" ]]; then
  echo "Expected MLIR CMake config at $(mlir_config_path)" >&2
  exit 1
fi

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
ADORA_JOBS="${ADORA_JOBS:-$(nproc)}"

if [[ ! -f "${BUILD_DIR}/build.ninja" ]]; then
  mkdir -p "${BUILD_DIR}"
  echo "Configuring ADORA in ${BUILD_DIR}"
  cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DMLIR_DIR="${LLVM_INSTALL_DIR}/lib/cmake/mlir" \
    -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
    -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

  # cmake -S "/data00/home/loujiahang/agent/aicb-agent/adora-compiler" -B "/data00/home/loujiahang/agent/aicb-agent/adora-compiler/build" -GNinja \
  #   -DCMAKE_INSTALL_PREFIX="/data00/home/loujiahang/agent/aicb-agent/adora-compiler/build" \
  #   -DCMAKE_BUILD_TYPE=Debug \
  #   -DMLIR_DIR="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/lib/cmake/mlir" \
  #   -DLLVM_BUILD_DIR="/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build" \
  #   -DLLVM_INSTALL_DIR="" \
  #   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  #   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Building ADORA (jobs=${ADORA_JOBS})..."
cmake --build "${BUILD_DIR}" --target install check-adora --parallel "${ADORA_JOBS}"
echo "Done."
