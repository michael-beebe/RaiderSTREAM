#!/bin/bash

ml load cmake

rm -rf build
mkdir -p build
cd build

# Test the --list flag
cmake \
  -DENABLE_OMP=ON \
  ../

make

export RS=bin/raiderstream

cd ../

./run.sh
