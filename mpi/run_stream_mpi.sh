#!/bin/bash

mpicc -fopenmp stream_mpi.c -o stream_mpi
export OMP_NUM_THREADS=1
mpirun -np 4 stream_mpi
rm stream_mpi
