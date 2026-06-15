#!/usr/bin/env bash
# Modified from:
# https://raw.githubusercontent.com/tensorflow/mlir-hlo/master/build_tools/build_mlir.sh
#
# One-shot LLVM/MLIR build. Override paths with environment variables:
#   LLVM_SRC_DIR       Path to llvm-project checkout (must contain llvm/)
#   LLVM_BUILD_DIR     CMake build directory (default: $LLVM_SRC_DIR/build)
#   LLVM_INSTALL_DIR   Install prefix (default: same as LLVM_BUILD_DIR)
#   CMAKE_BUILD_TYPE   Default: Debug (set Release for faster/smaller builds)
#   LLVM_JOBS          Parallel jobs (default: nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_llvm_src_dir() {
  if [[ -n "${LLVM_SRC_DIR:-}" ]]; then
    [[ -f "${LLVM_SRC_DIR}/llvm/CMakeLists.txt" ]] && return 0
    echo "LLVM_SRC_DIR=${LLVM_SRC_DIR} does not look like an llvm-project tree (missing llvm/CMakeLists.txt)." >&2
    exit 1
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
  echo "Could not find llvm-project. Clone/checkout commit b270525f730be6e7196667925f5a9bfa153262e9" >&2
  echo "and set LLVM_SRC_DIR to its root, e.g. export LLVM_SRC_DIR=/path/to/llvm-project-onnx" >&2
  exit 1
}

resolve_llvm_src_dir

LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SRC_DIR}/build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${LLVM_BUILD_DIR}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"
LLVM_JOBS="${LLVM_JOBS:-$(nproc)}"

echo "Using LLVM source dir: ${LLVM_SRC_DIR}"
echo "Build dir: ${LLVM_BUILD_DIR}"
echo "Install prefix: ${LLVM_INSTALL_DIR}"

mkdir -p "${LLVM_BUILD_DIR}"
mkdir -p "${LLVM_INSTALL_DIR}"

if [[ -f "${LLVM_BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "Existing CMake build in ${LLVM_BUILD_DIR}; reconfiguring is skipped (delete the dir to reconfigure)."
else
  echo "Configuring LLVM/MLIR..."
  set -x
  cmake -GNinja \
    "-H${LLVM_SRC_DIR}/llvm" \
    "-B${LLVM_BUILD_DIR}" \
    "-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_INCLUDE_TOOLS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_INCLUDE_TESTS=ON \
    -DMLIR_INCLUDE_TESTS=ON \
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}" \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_ENABLE_RTTI=ON \
    -DENABLE_LIBOMPTARGET=OFF \
    -DLLVM_ENABLE_LLD=OFF \
    -DBUILD_SHARED_LIBS=OFF
  set +x
fi

echo "Building and installing LLVM/MLIR targets (jobs=${LLVM_JOBS})..."
cmake --build "${LLVM_BUILD_DIR}" --parallel "${LLVM_JOBS}" \
  --target opt mlir-opt mlir-translate mlir-cpu-runner clang install

echo "Done. MLIR CMake package should be at: ${LLVM_INSTALL_DIR}/lib/cmake/mlir"


#### onnx version: v5.0.0 : https://github.com/onnx/onnx-mlir/tree/v0.5.0.0
#### LLVM FOR onnx:
### COMMIT b270525f730be6e7196667925f5a9bfa153262e9
### https://github.com/llvm/llvm-project/tree/b270525f730be6e7196667925f5a9bfa153262e9
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_dir" \
  -DCMAKE_INSTALL_PREFIX=$install_dir  \
  -DLLVM_INSTALL_UTILS=ON   \
  -DLLVM_ENABLE_PROJECTS="mlir;clang"   \
  -DLLVM_ENABLE_RUNTIMES="openmp"    \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV"   \
  -DLLVM_INCLUDE_TOOLS=ON   \
  -DLLVM_BUILD_TOOLS=ON   \
  -DLLVM_INCLUDE_TESTS=ON   \
  -DMLIR_INCLUDE_TESTS=ON   \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_RTTI=ON    \
 -DENABLE_LIBOMPTARGET=OFF \
  -DLLVM_ENABLE_LLD=OFF \
    -DBUILD_SHARED_LIBS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

 # MLIR_ENABLE_BINDINGS_PYTHON=ON enables the official mlir.ir / mlir.dialects
 # python packages (nanobind backend; run: pip install nanobind). This is an
 # in-place reconfigure of the existing build dir -- it only ADDS the python
 # binding targets, the already-built LLVM/MLIR libs are not invalidated.

 # TODO check what these options do :
  #  -DLLVM_ENABLE_LLD=ON   \
 # -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF 

cmake --build "$build_dir" --target opt mlir-opt mlir-translate mlir-cpu-runner clang install
# build the official MLIR python bindings package
cmake --build "$build_dir" --target MLIRPythonModules
ninja -j 72 install
