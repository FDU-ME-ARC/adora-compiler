#!/bin/bash
# This script was modified from:
# https://raw.githubusercontent.com/tensorflow/mlir-hlo/master/build_tools/build_mlir.sh

set -e

### set LLVM_SRC_DIR to your own path
LLVM_SRC_DIR=~/CGRVOPT/llvm-project-onnx

build_dir=$LLVM_SRC_DIR/build
install_dir=$LLVM_SRC_DIR/build

echo "Using LLVM source dir: $LLVM_SRC_DIR"

# Setup directories.
echo "Building MLIR in $build_dir"
mkdir -p "$build_dir"
echo "Creating directory to install: $install_dir"
mkdir -p "$install_dir"

echo "Beginning build (commands will echo)"
set -x


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