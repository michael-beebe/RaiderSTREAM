#!/bin/bash

root_dir=$(pwd)
exe=$root_dir/build/src/RaiderSTREAM/raiderstream

./clean.sh

mkdir -p build
cd build
rm -rf *

# Test the --help flag
# cmake -DDEBUG=1 -DENABLE_OMP=ON ../
# make -j4
# $exe --help ; echo ; echo ; echo

# Test the --list flag
cmake -DDEBUG=1 -DENABLE_OMP=ON ../
make -j4
$exe --list ; echo ; echo ; echo
