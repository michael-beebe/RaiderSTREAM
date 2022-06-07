#!/bin/bash
oshcc -fopenmp -DVERBOSE stream_openshmem.c -o stream_openshmem
export OMP_NUM_THREADS=1
oshrun -np 64 stream_openshmem
rm stream_openshmem