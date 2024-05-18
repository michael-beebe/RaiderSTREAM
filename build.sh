#!/bin/bash

ml load cmake

rm -rf build
mkdir -p build
cd build

# Test the --list flag
cmake \
  -DENABLE_OMP=ON \
  -DDEBUG=ON \
  ../

make

export RS=bin/raiderstream

cd ../


echo
./run.sh
