#!/bin/bash

rootfolder=$(pwd)
IRfolder="IR"
srcfolder="$rootfolder/$IRfolder/3_cgra_exes"
tarfolder="$rootfolder/$IRfolder/4_asms"
tempfolder="$rootfolder/$IRfolder/tempfiles"
func_name="jacobi-1d"

if [ ! -d "$tarfolder" ]; then
  mkdir -p "$tarfolder"
  echo mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi
cd $tempfolder
if [[ "$(pwd)" == "$tempfolder" ]]; then
  rm *.text
  cd -
fi
cd $tarfolder
if [[ "$(pwd)" == "$tarfolder" ]]; then
  rm *.s *.S
  cd -
fi

if [ -z "$CHIPYARD_DIR" ]; then
  echo "Environment variable CHIPYARD_DIR is not set. Please set it in env.sh and source env.sh."
  exit
fi

source $CHIPYARD_SOURCE_ENV
conda activate $CHIPYARD_DIR/.conda-env
# get syscalls.s crt.S
# riscv64-unknown-elf-gcc \
#  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
#  -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf \
#  -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
#  -lm -lgcc \
#  -I/home/tianyi/chipyard/generators/fdra/software/tests//riscv-tests \
#  -I/home/tianyi/chipyard/generators/fdra/software/tests//riscv-tests/env \
#  -I/home/tianyi/chipyard/generators/fdra/software/tests/ \
#  -I/home/tianyi/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common \
#  -DID_STRING=  -nostdlib -nostartfiles -static \
#  -T /home/tianyi/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common/test.ld \
#  -DBAREMETAL=1 \
#  -S -o $tarfolder/syscalls.s \
#  /home/tianyi/chipyard/generators/fdra/software/tests//riscv-tests/benchmarks/common/syscalls.c

cp  $CHIPYARD_DIR/generators/fdra/software/tests/gemm/crt.S $tarfolder/crt.s

include_line="#include \"include/ISA.h\""
cnt=0
# for file in "$srcfolder"/*.c; do
#     # Check whether the file is a regular file
#     filename=$(basename "$file" .c)
#     echo "$filename"
#     if [[ -f "$file" ]]; then
#       echo $cnt
#       ((cnt++))

#       if ! grep -q "$include_line" "$file"; then
#             temp_file="${file}.temp"
#             echo -e "$include_line\n$(cat "$file")" > "$temp_file"
#             mv "$temp_file" "$file"
#             echo "Added include directive to $file"
#       fi

#       # gcc .c to .asm
#       echo        riscv64-unknown-elf-gcc \
#         -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
#         -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf \
#         -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
#         -lm -lgcc \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/riscv-tests \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/riscv-tests/env \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/ \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/riscv-tests/benchmarks/common \
#         -DID_STRING=  -nostdlib -nostartfiles -static \
#         -DBAREMETAL=1 \
#         "$file" \
#         -S -o "$tarfolder/$filename.s"

#       riscv64-unknown-elf-gcc \
#         -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
#         -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf \
#         -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc12 \
#         -lm -lgcc \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/riscv-tests \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/riscv-tests/env \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/ \
#         -I${CHIPYARD_DIR}/generators/fdra/software/tests/riscv-tests/benchmarks/common \
#         -DID_STRING=  -nostdlib -nostartfiles -static \
#         -DBAREMETAL=1 \
#         "$file" \
#         -S -o "$tarfolder/$filename.s"

#       # if [[ -f "$tempfolder/cgra_execute.c" ]]; then
#       #   cp "$tempfolder/cgra_execute.c" $tarfolder/$filename.c
#       #   rm "$tempfolder/cgra_execute.c"
#       # else
#       #   echo "Mapping for $filename.json failed !"
#       #   exit 0
#       # fi
#     fi
# done
# cnt=0


# cd $rootfolder/$IRfolder

# echo "Using mlir-translate: "
# which mlir-translate
# echo "Using llvm opt: "
# which opt
# echo "Using llvm llc2 : "
# which llc

# ###
# # Host code on cpu
# ###
# # lowering only host to llvm
# echo mlir-opt -promote-buffers-to-stack --arith-expand --memref-expand  \
#  -normalize-memrefs --expand-strided-metadata  -lower-affine \
#  --scf-for-loop-canonicalization -convert-scf-to-cf \
#  --convert-math-to-llvm --convert-math-to-libm \
#  --convert-arith-to-llvm \
#   -normalize-memrefs  \
#    --import-constants-with-refs \
#  --finalize-memref-to-llvm="use-opaque-pointers" \
#   --import-constants-with-refs \
#    --memref-expand \
#  --finalize-memref-to-llvm="use-opaque-pointers" \
#  -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
#  -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
#  --finalize-memref-to-llvm="use-opaque-pointers" \
#  --cse --canonicalize \
#  --reconcile-unrealized-casts \
#  $rootfolder/$IRfolder/affine_host.mlir -o $rootfolder/$IRfolder/"3_${func_name}_llvm.mlir" \
#  --mlir-print-ir-after-all 2>&1 | cat > "3_intermediate_${func_name}_llvm.mlir"

# mlir-opt \
#  -promote-buffers-to-stack --arith-expand --memref-expand  \
#  --expand-strided-metadata  -lower-affine \
#  --scf-for-loop-canonicalization -convert-scf-to-cf \
#  --convert-math-to-llvm --convert-math-to-libm \
#  --convert-arith-to-llvm \
#   -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
#    --import-constants-with-refs \
#  --finalize-memref-to-llvm="use-opaque-pointers" \
#   --import-constants-with-refs \
#    --memref-expand \
#  --finalize-memref-to-llvm="use-opaque-pointers" \
#  -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
#  -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
#  --finalize-memref-to-llvm="use-opaque-pointers" \
#  --cse --canonicalize \
#  --reconcile-unrealized-casts \
#  $rootfolder/$IRfolder/affine_host.mlir -o $rootfolder/$IRfolder/"3_${func_name}_llvm.mlir" \
#  --mlir-print-ir-after-all 2>&1 | cat > "3_intermediate_${func_name}_llvm.mlir"

#   # -normalize-memrefs  \
#   #  --mlir-elide-elementsattrs-if-larger=2 
#  #  --adora-convert-kernelcall-to-llvm \

# # mlir-translate  --mlir-to-llvmir $rootfolder/$IRfolder/"3_${func_name}_llvm.mlir" > "$rootfolder/$IRfolder/$func_name.ll"

# mlir-translate  --mlir-to-llvmir \
#  $rootfolder/$IRfolder/"3_${func_name}_llvm.mlir" \
#  -o "$rootfolder/$IRfolder/$func_name.ll"


# # opt -memprof  "$rootfolder/$IRfolder/$func_name.ll" -o "$rootfolder/$IRfolder/$func_name.bc"
# # echo opt -O3 \
# #  --disable-builtin=memset "$rootfolder/$func_name.ll" -o "$rootfolder/$func_name.bc"
# # opt -O3 \
# #  --disable-builtin=memset "$rootfolder/$IRfolder/$func_name.ll" \
# #  -o "$rootfolder/$IRfolder/$func_name.bc"

# opt -O3  "$rootfolder/$IRfolder/$func_name.ll" --disable-builtin=memset  --mtriple=riscv64 --mcpu=rocket-rv64 -o "$rootfolder/$IRfolder/$func_name.bc"
# opt -O3  "$rootfolder/$IRfolder/$func_name.ll" --disable-builtin=memset  --mtriple=riscv64 --mcpu=rocket-rv64 -S -o "$rootfolder/$IRfolder/${func_name}_opt.ll"

# # echo llc -O3 "$rootfolder/$func_name.bc" \
# #   -I /home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include\
# #   -march=riscv64 -mtriple=riscv64-unknown-elf-gnu -mcpu=rocket-rv64 \
# #   -mattr=+c,+d,+relax,+m  \
# #   --relocation-model=pic \
# #   -float-abi=hard \
# #   -code-model=small \
# #   -o "$tarfolder/$func_name.s"
# llc -O3 "$rootfolder/$IRfolder/$func_name.bc" \
#   -I /home/jhlou/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/include\
#   -march=riscv64 -mtriple=riscv64-unknown-elf-gnu -mcpu=rocket-rv64 \
#   -mattr=+c,+d,+relax,+m  \
#   --relocation-model=pic \
#   -float-abi=hard \
#   -code-model=small \
#   -o "$tarfolder/$func_name.s"