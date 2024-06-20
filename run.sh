#!/bin/bash

cd ./build/bin

# --- OpenMP Test
#export OMP_NUM_THREADS=20
./raiderstream -k all -s 1000000 -np 1

# --- OpenACC Test  
# ./raiderstream -k seq_copy -s 200000000 -np 1

# --- MPI/OpenMP Test
# mpirun -np 4 --mca btl ^openib raiderstream -k all -s 10000000 -np 4

# --- OpenSHMEM/OpenMP Test
# export UCX_LOG_LEVEL=debug
# ulimit -a
# ulimit -m unlimited
# ulimit -v unlimited
# ulimit -n 4096
# export SHMEM_SYMMETRIC_HEAP_SIZE=10G
# export SHMEM_MAX_SEGMENTS=128
# oshrun --mca btl ^openib --mca opal_common_ucx_opal_mem_hooks 1 -np 1 ./raiderstream -k seq_copy -s 100000 -np 1
