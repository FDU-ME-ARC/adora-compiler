#!/bin/bash

# Specify the folder path
rootfolder=$(pwd)
IRfolder="IR"
srcfolder="$rootfolder/$IRfolder/3_cgra_exes"
asmfolder="$rootfolder/$IRfolder/4_asms"
tarfolder="$rootfolder/$IRfolder/5_obj"
# tarfolder="$CHIPYARD_DIR/generators/fdra/software/tests/bareMetalC"
tempfolder="$rootfolder/$IRfolder/tempfiles"
main_file="gemm_main.c"
baremetal_file_name="gemm_mini"
DATASET_Size="MINI_DATASET"

if [ ! -d "$tarfolder" ]; then
  mkdir -p "$tarfolder"
  echo mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi

if [ -z "$CHIPYARD_DIR" ]; then
  echo "Environment variable CHIPYARD_DIR is not set. Please set it in env.sh and source env.sh."
  exit
fi

source $CHIPYARD_SOURCE_ENV
conda activate $CHIPYARD_DIR/.conda-env

set -x

# asm to object
asm_files=$(find "$asmfolder" -name "*.s" -type f)
src_files=$(find "$srcfolder" -name "*.c" -type f)

riscv64-unknown-elf-gcc \
 -DROCKET_TARGET -D${DATASET_Size} \
 -Wl,--wrap,malloc \
 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
 -O2 -ffast-math -fno-common -fno-builtin-printf \
 -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
 -lm \
 -I$CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/benchmarks/common \
 -I$CHIPYARD_DIR/generators/fdra/software/tests/riscv-tests \
 -I$CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/env \
 -I$CHIPYARD_DIR/generators/fdra/software/tests/ \
  -I/home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities \
 -T$CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/my_test.ld \
 -DID_STRING= -nostartfiles -static  -DDEFINE_MALLOC \
 -DBAREMETAL=1 -e _start -g  \
 -o $tarfolder/${baremetal_file_name}-baremetal \
 $rootfolder/$main_file \
 /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/medley/deriche/deriche_mini/cpu_deriche.c \
 $src_files  $asm_files\
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/syscalls.c \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp \
 $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c


cp $tarfolder/${baremetal_file_name}-baremetal $CHIPYARD_DIR/generators/fdra/software/tests/build/bareMetalC/

set +x

#  -lgcc  -fno-builtin-malloc -fno-builtin-free  -fno-builtin \
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c \
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp 
#  /home/jhlou/CGRVOPT/cgra-opt/experiment/Cbenchmarks/Polybench/utilities/rocket_polybench.c
#  $CHIPYARD_DIR/generators/fdra/software/tests//riscv-tests/benchmarks/common/syscalls.c \
#  $CHIPYARD_DIR/generators/fdra/software/tests/gemm/crt.S \
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/CRunnerUtils.cpp
#  $CHIPYARD_DIR/generators/fdra/software/tests/UtilSrc/tiny-malloc.c 