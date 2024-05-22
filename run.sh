#!/bin/bash

cd ./build/bin

# --- OpenMP Test
#./raiderstream -k all -s 10000000 -np 1

# --- MPI/OpenMP Test
export OMP_NUM_THREADS=10
mpirun -np 4 --mca btl ^openib raiderstream -k all -s 10000000 -np 4

