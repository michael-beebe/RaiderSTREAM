#!/bin/bash


# -------------------------------------------
# Single core run for each implementation
# OMP_NUM_THREADS=0
# -np 1
# -------------------------------------------
NP_VALUE=1
export STREAM_ARRAY_SIZE=2500000
STREAM_DIR=.


#------------------------------------------------------------
# Setting vars for file paths to each STREAM implementation
#------------------------------------------------------------
export ORIGINAL_IMPL=$STREAM_DIR/stream_original.c
export OMP_IMPL=$STREAM_DIR/openmp/stream_openmp.c
export SHEM_IMPL=$STREAM_DIR/openshmem/stream_openshmem.c
export MPI_IMPL=$STREAM_DIR/mpi/stream_mpi.c


module purge
module load gcc
module load openmpi


make all
echo "====================================================================================="
echo "               STREAM BENCHMARK RUN ON "$(date +"%d-%m-%y")" AT "$(date +"%T")
echo "====================================================================================="
echo "-------------------------------------------------------------"
echo "                      'Original' STREAM"
echo "-------------------------------------------------------------"
if ./stream_original.exe; then
        echo "Original impl finished."
else
	echo "Original impl FAILED!"
fi
echo ""
echo ""


echo "-------------------------------------------------------------"
echo "                           OpenMP"
echo "-------------------------------------------------------------"
if ./stream_omp.exe; then
        echo "OpenMP impl finished."
else
        echo "OpenMP impl FAILED!"
fi
echo ""
echo ""


echo "-------------------------------------------------------------"
echo "                             MPI"
echo "-------------------------------------------------------------"
if mpirun -np $NP_VALUE stream_mpi.exe; then
        echo "MPI impl finished."
else
        echo "MPI impl FAILED!"
fi
echo ""
echo ""


echo "-------------------------------------------------------------"
echo "                          OpenSHMEM"
echo "-------------------------------------------------------------"
if oshrun -np $NP_VALUE stream_oshmem.exe; then
        echo "OpenSHMEM impl finished."
else
        echo "OpenSHMEM impl FAILED!"
fi


make clean
