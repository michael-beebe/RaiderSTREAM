#!/bin/bash


# -------------------------------------------
# Slurm setup
# -------------------------------------------
#SBATCH -J STREAM_MULTI_TESTING
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:05:00

# --ntasks-per-node max is 128 1 task is 1 core
# --ntasks or -n is total task (not per node) total cores



# -------------------------------------------
# Single core run for each implementation
# OMP_NUM_THREADS=0
# -np 1
# -------------------------------------------
export STREAM_ARRAY_SIZE=2500000
export OMP_NUM_THREADS=1
NP_VALUE=4



#------------------------------------------------------------
# Load correct modules
#------------------------------------------------------------
module purge
module load gcc
module load openmpi
module list



#------------------------------------------------------------
# For all of the following make call you can add PFLAGS and CFLAGS.
# If you are running only on version of stream or the same flags between
# the different versions of stream you can use export CFLAGS or export PFLAGS
# to define the variable.

# Use PFLAGS for program flags ie -DVERBOSE and -DDEBUG
# Use CFLAGS for compiler flags such as -fopenmp
#------------------------------------------------------------
make stream_original.exe PFLAGS="-DVERBOSE"
make stream_omp.exe CFLAGS="-fopenmp" PFLAGS="-DVERBOSE"
make stream_mpi.exe CFLAGS="-fopenmp" PFLAGS="-DVERBOSE"
make stream_oshmem.exe CFLAGS="-fopenmp" PFLAGS="-DVERBOSE"



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
