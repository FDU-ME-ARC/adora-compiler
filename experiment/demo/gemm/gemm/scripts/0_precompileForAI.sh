######
# C to mlir affine
######
rootfolder=$(pwd)
IRfolder=$rootfolder/"IR"
kernel_src=affine
tarfolder="$IRfolder/0_kernels"

if [ -d "$IRfolder" ]; then
  rm -r previous_IR
  mv $IRfolder previous_IR
fi

echo mkdir -p "$IRfolder"
mkdir -p "$IRfolder"

# if [ ! -d "$tarfolder" ]; then
#   mkdir -p "$tarfolder"
# fi

if [ ! -d "$CHIPYARD_DIR" ]; then
    echo "CHIPYARD_DIR does not set. Please source cgra-opt/env.sh first."
    exit
fi

if [ ! -d "$CGRVOPT_PROJECT_PATH" ]; then
    echo "CGRVOPT_PROJECT_PATH does not set. Please source cgra-opt/env.sh first."
    exit
fi


mlir-opt\
 --allow-unregistered-dialect      \
 --affine-loop-normalize \
 --affine-simplify-structures   \
 --normalize-memrefs \
 --force-specialization \
 --bufferization-bufferize \
 $kernel_src.mlir \
 -o  $IRfolder/"$kernel_src"_normalized.mlir



##################
### Extract affine for kernels
##################
cd $IRfolder
cgra-opt \
    --canonicalize \
    -reconcile-unrealized-casts \
    --affine-loop-fusion \
    --adora-extract-affine-for-to-kernel \
    --arith-expand --memref-expand \
    -cse \
    --adora-extract-kernel-to-function="kernel-gen-dir=$IRfolder" \
    $IRfolder/"$kernel_src"_normalized.mlir -o $IRfolder/"$kernel_src"_host.mlir

        # --adora-extract-kernel-to-function="kernel-gen-dir=$IRfolder" \
mv $IRfolder/kernels $tarfolder

cd -
if [ $? -eq 0 ]; then
    echo "Kernels for CGRA are recognized."
fi