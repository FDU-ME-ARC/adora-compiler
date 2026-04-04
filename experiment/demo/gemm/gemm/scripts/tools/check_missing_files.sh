#!/bin/bash
# folder_path="/home/jhlou/CGRVOPT/cgra-opt/experiment/AIModels/bert/IR/tempfiles/DFGs" 
# files=$(ls "$folder_path" | grep -oP "forward_kernel_\d+_CDFG")
# file_numbers=($(echo "$files" | sed -E 's/forward_kernel_([0-9]+)_CDFG/\1/' | sort -n))

folder_path="/home/jhlou/CGRVOPT/cgra-opt/experiment/AIModels/bert/IR/3_cgra_exes" 
files=$(ls "$folder_path" | grep -oP "forward_kernel_\d+_exe")
file_numbers=($(echo "$files" | sed -E 's/forward_kernel_([0-9]+)_exe/\1/' | sort -n))


missing_files=()
for ((i=${file_numbers[0]}; i<=${file_numbers[-1]}; i++)); do
  if [[ ! " ${file_numbers[@]} " =~ " $i " ]]; then
    missing_files+=("forward_kernel_${i}_exe")
  fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
  echo "no missing files"
else
  echo "missing files:"
  for missing in "${missing_files[@]}"; do
    echo "$missing"
  done
fi
