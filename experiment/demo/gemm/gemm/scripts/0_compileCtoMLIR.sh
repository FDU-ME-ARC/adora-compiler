######
# C to mlir affine
######
rootfolder=$(pwd)
IRfolder="IR"
kernel_src=gemm
DATASET_Size="MINI_DATASET"
tarfolder="$rootfolder/$IRfolder/0_kernels"

if [ -d "$IRfolder" ]; then
  mv $IRfolder previous_IR
fi

echo mkdir -p "$IRfolder"
mkdir -p "$IRfolder"

if [ ! -d "$tarfolder" ]; then
  mkdir -p "$tarfolder"
fi

if [ ! -d "$CHIPYARD_DIR" ]; then
    echo "CHIPYARD_DIR does not set. Please source cgra-opt/env.sh first."
    exit
fi

if [ ! -d "$CGRVOPT_PROJECT_PATH" ]; then
    echo "CGRVOPT_PROJECT_PATH does not set. Please source cgra-opt/env.sh first."
    exit
fi



##################
### Compile C to MLIR
##################
clear && \
cgeist \
    -O2 \
    -lm -lgcc \
    -Dsize_t=int -Dwint_t=int -DROCKET_TARGET -D_riscv -DDATA_TYPE_IS_FLOAT \
    -D${DATASET_Size} \
    --import-all-index \
    -I$CGRVOPT_PROJECT_PATH/experiment/Cbenchmarks/Polybench/utilities \
    -I$CHIPYARD_DIR/.conda-env/riscv-tools/riscv64-unknown-elf/include/machine/ \
    -I$CHIPYARD_DIR/.conda-env/riscv-tools/sysroot/usr/include/linux/ \
   $kernel_src.c \
    -S -o $IRfolder/$kernel_src.mlir

# -I$CHIPYARD_DIR/.conda-env/riscv-tools/riscv64-unknown-elf/include/ \
# -I$CHIPYARD_DIR/.conda-env/riscv-tools/lib/gcc/riscv64-unknown-elf/12.2.0/include \
# -I$FDRA_DIR/software/tests \
# -I$FDRA_DIR/software/tests/riscv-tests/env \
# -I$FDRA_DIR/software/tests/riscv-tests \
# -I$FDRA_DIR/software/tests/riscv-tests/benchmarks/common \

mlir-opt\
 --allow-unregistered-dialect      \
 --affine-loop-normalize \
 --affine-simplify-structures   \
 --normalize-memrefs \
 --force-specialization \
 --bufferization-bufferize \
 $IRfolder/$kernel_src.mlir \
 -o  $IRfolder/"$kernel_src"_normalized.mlir



##################
### Extract affine for kernels
##################
cgra-opt \
    --canonicalize \
    -reconcile-unrealized-casts \
    --affine-loop-fusion \
    --adora-extract-affine-for-to-kernel \
    --arith-expand --memref-expand \
    -cse \
    $IRfolder/"$kernel_src"_normalized.mlir -o $tarfolder/"$kernel_src"_kernel.mlir

        # --adora-extract-kernel-to-function="kernel-gen-dir=$IRfolder" \

if [ $? -eq 0 ]; then
    echo "Kernels for CGRA are recognized."
fi