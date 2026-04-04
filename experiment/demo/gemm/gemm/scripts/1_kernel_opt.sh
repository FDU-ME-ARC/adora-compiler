#!/bin/bash
###################
## User define args 
###################
unroll=0 # 1: auto unroll, 0: Don't unroll
###################
## Don't touch other part
###################

rootfolder=$(pwd)
IRfolder="IR"
srcfolder="$rootfolder/$IRfolder/0_kernels"
# srcfolder="$rootfolder/$IRfolder/extra"
tarfolder="$rootfolder/$IRfolder/1_kernels_opt"
# tarfolder="$rootfolder/$IRfolder/extra_opt"
tempfolder="$rootfolder/$IRfolder/tempfiles"

echo "current path:$rootfolder"

if [ ! -d "$tarfolder" ]; then
  echo mkdir -p "$tarfolder"
  mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi

cd $tarfolder 
rm *.mlir

cd $tempfolder
echo "current path:$tempfolder"
# traverse every file
cnt=0

if [ -z "$CGRA_ADG_PATH" ]; then
  echo "Environment variable CGRA_ADG_PATH is not set. Please set it in env.sh and source env.sh."
  exit
fi

for file in "$srcfolder"/*; do
    filename=$(basename "$file" .mlir)
    echo "$filename"
    if [[ -f "$file" ]]; then
      if [[ ${unroll} -eq 1 ]]; then
        cgra-opt \
          --adora-simplify-loadstore \
          --adora-math-rewrite \
          --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
          --adora-auto-unroll="cgra-adg=${CGRA_ADG_PATH}/cgra_adg.json" \
          --adora-schedule-tasks \
          "$file" -o $tarfolder/"$filename"_opt.mlir
      else
         cgra-opt \
          --adora-simplify-loadstore \
          --adora-math-rewrite \
          --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
          --adora-schedule-tasks \
          "$file" -o $tarfolder/"$filename"_opt.mlir
      fi

        # 
        # --adora-simplify-loadstore \
        # --adora-extract-affine-for-to-kernel \
        # --adora-hoist-loadstore \
 # --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
        # --adora-loop-unroll-jam \
        # --adora-affine-loop-unroll="cgra-adg=${CGRA_ADG_PATH}/cgra_adg.json" \
        # 
        # --adora-hoist-loadstore \
        # "$file" -o $tarfolder/"$filename"_opt.mlir
      ((cnt++))
      echo $cnt
    fi
done

cd $rootfolder