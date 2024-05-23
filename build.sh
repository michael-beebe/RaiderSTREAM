#!/bin/bash

# --- Load required modules
module load cmake

# --- Remove and recreate the build directory
rm -rf build
mkdir -p build
cd build

# --- Configure the project with CMake
cmake                                   \
  -DENABLE_SHMEM_OMP=ON                 \
  -DCMAKE_C_COMPILER=`which oshcc`      \
  -DCMAKE_CXX_COMPILER=`which oshc++`   \
  -DCMAKE_C_FLAGS="-fopenmp"            \
  -DCMAKE_CXX_FLAGS="-fopenmp"          \
  ../



# cmake \
#   -DENABLE_MPI_OMP=ON \
#   -DCMAKE_C_COMPILER=`which mpicc` \
#   -DCMAKE_CXX_COMPILER=`which mpic++` \
#   ../



# cmake \
#   -DENABLE_OMP=ON \
#  ../

# --- Build the project
make

# --- Set the path to the executable
export RS=bin/raiderstream

# --- Go back to the project root directory
cd ../

# --- Run the executable
echo ; ./run.sh ; echo ; echo

