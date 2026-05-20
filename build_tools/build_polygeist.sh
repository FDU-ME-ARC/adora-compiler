#!/bin/bash
# One-shot build for the Polygeist (C/C++ -> MLIR) frontend.
#
# Polygeist requires its OWN LLVM commit (26eb4285...), which is different from
# the LLVM (b270525f...) shared by adora-compiler and adora-onnx-mlir.
# Do NOT reuse $HOME/CGRVOPT/llvm-project-onnx for Polygeist.
#
# Steps:
#   0. Ensure the Polygeist submodule and its bundled llvm-project sub-submodule are checked out.
#   1. Build Polygeist's private LLVM/MLIR/Clang at frontend/Polygeist/llvm-project/build.
#   2. Build & install Polygeist itself at frontend/Polygeist/build.
#
# Re-running this script is safe: cmake is skipped if CMakeCache.txt exists,
# so subsequent invocations become incremental ninja builds.
#
# Env overrides:
#   JOBS=N  parallel jobs (default: nproc)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POLYGEIST_SRC="${PROJECT_ROOT}/frontend/Polygeist"
LLVM_SRC="${POLYGEIST_SRC}/llvm-project"
LLVM_BUILD="${LLVM_SRC}/build"
POLYGEIST_BUILD="${POLYGEIST_SRC}/build"

JOBS="${JOBS:-$(nproc)}"

echo "[polygeist] PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "[polygeist] POLYGEIST_SRC    = ${POLYGEIST_SRC}"
echo "[polygeist] LLVM_SRC         = ${LLVM_SRC}"
echo "[polygeist] LLVM_BUILD       = ${LLVM_BUILD}"
echo "[polygeist] POLYGEIST_BUILD  = ${POLYGEIST_BUILD}"
echo "[polygeist] JOBS             = ${JOBS}"

# ---------- step 0: ensure submodules are present ----------
if [ ! -f "${POLYGEIST_SRC}/CMakeLists.txt" ]; then
  echo "[polygeist] frontend/Polygeist not initialized — running 'git submodule update --init --recursive frontend/Polygeist'"
  git -C "${PROJECT_ROOT}" submodule update --init --recursive frontend/Polygeist
fi
if [ ! -f "${LLVM_SRC}/llvm/CMakeLists.txt" ]; then
  echo "[polygeist] Polygeist's llvm-project sub-submodule missing — initializing recursively"
  git -C "${POLYGEIST_SRC}" submodule update --init --recursive
fi

# ---------- step 1: build Polygeist's private LLVM (mlir + clang) ----------
echo "[polygeist] === Step 1/2: building private LLVM at ${LLVM_BUILD} ==="
mkdir -p "${LLVM_BUILD}"
if [ ! -f "${LLVM_BUILD}/CMakeCache.txt" ]; then
  cmake -GNinja \
    "-H${LLVM_SRC}/llvm" \
    "-B${LLVM_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${LLVM_BUILD}" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_INCLUDE_TOOLS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_INCLUDE_TESTS=ON \
    -DMLIR_INCLUDE_TESTS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DLLVM_ENABLE_RTTI=ON \
    -DENABLE_LIBOMPTARGET=OFF \
    -DLLVM_ENABLE_LLD=OFF \
    -DLLVM_USE_LINKER=gold \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DBUILD_SHARED_LIBS=OFF
fi
ninja -C "${LLVM_BUILD}" -j "${JOBS}" install

# ---------- step 2: build Polygeist itself ----------
echo "[polygeist] === Step 2/2: building Polygeist at ${POLYGEIST_BUILD} ==="
mkdir -p "${POLYGEIST_BUILD}"
if [ ! -f "${POLYGEIST_BUILD}/CMakeCache.txt" ]; then
  cmake -GNinja \
    "-H${POLYGEIST_SRC}" \
    "-B${POLYGEIST_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${POLYGEIST_BUILD}" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" \
    -DCLANG_DIR="${LLVM_BUILD}/lib/cmake/clang" \
    -DLLVM_EXTERNAL_LIT="${LLVM_BUILD}/bin/llvm-lit" \
    -DBUILD_SHARED_LIBS=OFF \
    -DPOLYGEIST_ENABLE_CUDA=0 \
    -DPOLYGEIST_ENABLE_ROCM=0 \
    -DPOLYGEIST_ENABLE_POLYMER=0
fi
ninja -C "${POLYGEIST_BUILD}" -j "${JOBS}" install

echo "[polygeist] === Done ==="
echo "  cgeist:        ${POLYGEIST_BUILD}/bin/cgeist"
echo "  polygeist-opt: ${POLYGEIST_BUILD}/bin/polygeist-opt"
