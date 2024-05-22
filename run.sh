#!/bin/bash

cd ./build/bin

# --- OpenMP Test
export OMP_NUM_THREADS=10
# ./raiderstream -k all -s 10000000 -np 0

# --- MPI/OpenMP Test
# mpirun -np 4 --mca btl ^openib raiderstream -k all -s 10000000 -np 4

# --- OpenSHMEM/OpenMP Test
oshrun -np 4 --mca btl ^openib raiderstream -k all -s 10000000 -np 4
