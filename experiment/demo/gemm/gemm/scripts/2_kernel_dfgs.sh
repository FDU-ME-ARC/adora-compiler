#!/bin/bash

rootfolder=$(pwd)
IRfolder="IR"
srcfolder="$rootfolder/$IRfolder/1_kernels_opt"
tarfolder="$rootfolder/$IRfolder/2_dfgs"
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

cd $dfgfolder
rm *

cd $tarfolder
if [[ "$(pwd)" == "$tarfolder" ]]; then
  rm *.dot
  cd -
fi

cd $rootfolder
# traverse every file
cnt=0

cd $tempfolder/$dfgfolder
for file in "$srcfolder"/*.mlir; do
    filename=$(basename "$file" _opt.mlir)
    echo file : "$filename"
    if [[ -f "$file" ]]; then
      cgra-opt\
        --adora-kernel-dfg-gen \
        $file 
    
      # ((cnt++))
      # echo $cnt
    fi
done

for file in "$tempfolder/$dfgfolder"/*_CDFG.dot; do
    filename=$(basename "$file" _CDFG.dot)
    echo dfg $cnt: "$filename"
    if [[ -f "$file" ]]; then
      
      cp ./"$filename"_CDFG.dot $tarfolder/"$filename"_CDFG.dot
      dot $tarfolder/"$filename"_CDFG.dot -Tpng -o $tarfolder/"$filename"_CDFG.png

      ((cnt++))
      # echo $cnt
    fi
done
cnt=0
cd -
