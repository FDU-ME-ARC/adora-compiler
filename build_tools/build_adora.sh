#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

### set LLVM_BUILD_DIR to your own llvm path
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$HOME/CGRVOPT/llvm-project-onnx/build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${LLVM_BUILD_DIR}}"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake -GNinja \
      "$PROJECT_ROOT" \
      -DCMAKE_INSTALL_PREFIX=. \
      -DCMAKE_BUILD_TYPE=Debug \
      -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
      -DLLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
      -DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
else
    cd "$BUILD_DIR"
fi

ninja -j $(nproc) install check-adora
# or ninja -j 32 install