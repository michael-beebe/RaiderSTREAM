#!/bin/bash

./clean.sh

mkdir -p build
cd build
rm -rf *
cmake -DENABLE_OMP=ON ../

make

