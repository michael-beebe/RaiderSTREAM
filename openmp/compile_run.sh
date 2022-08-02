#!/bin/bash
gcc -fopenmp stream_openmp.c -o stream_omp
export OMP_NUM_THREADS=4
./stream_omp
rm stream_omp
