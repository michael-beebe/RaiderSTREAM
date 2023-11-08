#!/bin/bash

mkdir -p build
cd build
rm -rf *
cmake -DENABLE_OMP=ON ../

make

