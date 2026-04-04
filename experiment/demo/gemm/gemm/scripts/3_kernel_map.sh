#!/bin/bash

rootfolder=$(pwd)
IRfolder="IR"
srcfolder="$rootfolder/$IRfolder/1_kernels_opt"
tarfolder="$rootfolder/$IRfolder/3_cgra_exes"
tempfolder="$rootfolder/$IRfolder/tempfiles"
# kernel_basename="forward_kernel"

if [ ! -d "$tarfolder" ]; then
  mkdir -p "$tarfolder"
  echo mkdir -p "$tarfolder"
fi
if [ ! -d "$tempfolder" ]; then
  mkdir -p "$tempfolder"
  echo mkdir -p "$tempfolder"
fi

cd $tempfolder
# if [[ "$(pwd)" == "$tempfolder" ]]; then
#   rm *.dot *.ll *.json *.txt
#   cd -
# fi

dfgfolder="DFGs" 
if [ ! -d "$dfgfolder" ]; then
  mkdir -p "$dfgfolder"
  echo mkdir -p "$dfgfolder"
fi

cd $tarfolder
if [[ "$(pwd)" == "$tarfolder" ]]; then
  rm *.dot
  cd -
fi
cd $dfgfolder
rm *

cd $rootfolder
# traverse every file
cnt=0

if [ -z "$CGRA_ADG_PATH" ]; then
  echo "Environment variable CGRA_ADG_PATH is not set. Please set it in env.sh and source env.sh."
  exit
fi
if [ -z "$CGRA_OP_FILE_PATH" ]; then
  echo "Environment variable CGRA_OP_FILE_PATH is not set. Please set it in env.sh and source env.sh."
  exit
fi

cd $tempfolder

## single thread
# for file in "$srcfolder"/*.mlir; do
#     filename=$(basename "$file" _opt.mlir)
#     echo "$filename"
#     if [[ -f "$file" ]]; then
#       mkdir -p "$tempfolder/map_result_${cnt}"
#       cgra-mapper \
#         --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
#         --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
#         --output="$tempfolder/map_result_${cnt}/cgra_exe.c" \
#         $file 
      
#       cp "$tempfolder/map_result_${cnt}/cgra_exe.c" $tarfolder/"$filename"_exe.c

#       ((cnt++))
#       echo $cnt
#     fi
# done
# cnt=0

MAX_THREADS=8
current_jobs=0

for file in "$srcfolder"/*.mlir; do
    filename=$(basename "$file" _opt.mlir)
    echo "$filename"
    if [[ -f "$file" ]]; then
        # control prallel tasks
        while (( current_jobs >= MAX_THREADS )); do
            wait -n
            ((current_jobs--))
        done

        # command execute
        (
            mkdir -p "$tempfolder/map_result_${cnt}"
            cgra-mapper \
                --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
                --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
                --output="$tempfolder/map_result_${cnt}/${filename}_exe.c" \
                "$file" 
            
            cp "$tempfolder/map_result_${cnt}/${filename}_exe.c" "$tarfolder/$filename"_exe.c
            ((cnt++))
            echo "Processed $filename, cnt: $cnt"
        ) &

        ((current_jobs++))  # increase onflight task
    fi
done

# wait all tasks
wait
echo "All tasks completed!"

cd -
