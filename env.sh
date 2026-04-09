
##########################
#### You need to change the following 3 paths to your own local path
#### If you don't want need cgra hardware to execute simulation, CHIPYARD_SOURCE_ENV can be omit.
######################### 
export CGRA_ADG_PATH="/home/jhlou/chipyard/generators/fdra/cgra-mg/src/main/resources"
export CHIPYARD_DIR="/home/jhlou/chipyard"

##########################
#### ADORA projects should be placed at chipyard/generators
######################### 
export ADORA_DIR=$CHIPYARD_DIR"/generators/fdra"

##########################
#### Don't have to change paths beneath
######################### 
export ADORA_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CGRA_OP_FILE_PATH=$CGRA_ADG_PATH
export GeneralOpNameFile="$ADORA_PROJECT_PATH/lib/DFG/Documents/GeneralOpName.txt"

export PATH=${ADORA_PROJECT_PATH}/build/bin:$PATH
export PATH=${ADORA_PROJECT_PATH}/frontend/adora-onnx-mlir/build/bin:$PATH

####### CONDA IN CHIPYARD
export CHIPYARD_SOURCE_ENV="$CHIPYARD_DIR/env.sh"
source $CHIPYARD_SOURCE_ENV
conda activate $CHIPYARD_DIR/.conda-env

export LLVM_PROJ_BUILD="/home/jhlou/CGRVOPT/llvm-project-onnx/build"

# Dynamically locate the copied protobuf build directory
export PROTOBUF_BUILD_DIR="${ADORA_PROJECT_PATH}/frontend/adora-onnx-mlir/third_party/protobuf/build"

if [ -d "$PROTOBUF_BUILD_DIR" ]; then
    export PATH="${PROTOBUF_BUILD_DIR}/bin:$PATH"
    export LD_LIBRARY_PATH="${PROTOBUF_BUILD_DIR}/lib:$LD_LIBRARY_PATH"
else
    echo "[Env] Warning: Local Protobuf not found at $PROTOBUF_BUILD_DIR"
fi

if [ -d "/home/share/llvm-project-Polygeist/build/bin" ]; then
    export PATH="/home/share/llvm-project-Polygeist/build/bin:$PATH"
fi

export PATH="/opt/cmake-3.31.1/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export PATH=$PATH:/home/ykchen/.local/bin

export PATH=/home/jhlou/zwzhong/verilator-master/install/usr/local/bin:$PATH