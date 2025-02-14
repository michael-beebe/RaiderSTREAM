#!/bin/bash

#     Make sure script has correct number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <gpu> <implementation>"
  echo "Where <gpu> is either v100 or a100" #Or h100
  echo "And <implementation> is either openmp, openacc, or cuda"
  exit 1
fi

#     Assign arguments to variables
GPU=$1
IMPL=$2

#     Validate GPU argument
if [[ "$GPU" != "v100" && "$GPU" != "a100" ]]; then
  echo "Invalid GPU specified. Argument 1 must be 'a100' or 'v100'" #or 'h100'
  exit 1
fi
#     Validate the Backend arguement
if [[ "$IMPL" != "openmp" && "$IMPL" != "openacc" && "$IMPL" != "cuda" ]]; then
  echo "Invalid backend specified. Must be 'openmp', 'openacc', or 'cuda'."
  exit 1
fi

#     Print arguements
echo "GPU: $GPU"
echo "IMPL: $IMPL"

#     Create variables to hold compilation flags
ENABLE_FLAG=""
C_COMPILER=""
C_FLAGS=""
CXX_COMPILER=""
CXX_FLAGS=""
EXE_LINKER_FLAGS=""

#     Set flags based on GPU and implementation
if [[ "$GPU" == "v100" ]]; then
  case "$IMPL" in
    openmp)
      ml load gcc/10.1.0
      ml load nvhpc/21.3-mpi
      ENABLE_FLAG="-DENABLE_OMP_TARGET=ON"
      ;;
    openacc)
      ml load gcc/10.1.0
      ml load nvhpc/21.3-mpi
      ENABLE_FLAG="-DENABLE_OACC=ON"
      C_COMPILER=`which nvc`
      CXX_COMPILER=`which nvc++`
      C_FLAGS="-acc -ta=tesla:cc70 -Minfo=accel"
      CXX_FLAGS=$C_FLAGS
      ;;
    cuda)
      ml load gcc/8.4.0
      ml load cuda/10.2.89
      ENABLE_FLAG="-DENABLE_CUDA=ON"
      C_COMPILER=`which nvcc`
      CXX_COMPILER=`which nvcc`
      EXE_LINKER_FLAGS="-lcudart -lcudaevrt"
      ;;
  esac
fi

echo "ENABLE_FLAG: $ENABLE_FLAG"
echo "C_COMPILER: $C_COMPILER"
echo "CXX_COMPILER: $CXX_COMPILER"
echo "C_FLAGS: $C_FLAGS"
echo "CXX_FLAGS: $CXX_FLAGS"

cd ../
rm -rf build
mkdir build
cd build
cmake $ENABLE_FLAG -DCMAKE_CXX_COMPILER=$CXX_COMPILER -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_FLAGS=$CXX_FLAGS -DCMAKE_C_FLAGS=$C_FLAGS -DCMAKE_EXE_LINKER_FLAGS=$EXE_LINKER_FLAGS ../
make
