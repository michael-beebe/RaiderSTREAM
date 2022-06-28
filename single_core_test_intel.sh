#!/bin/bash


# Architecture:        x86_64
# CPU op-mode(s):      32-bit, 64-bit
# Byte Order:          Little Endian
# CPU(s):              128
# On-line CPU(s) list: 0-127
# Thread(s) per core:  1
# Core(s) per socket:  64
# Socket(s):           2
# NUMA node(s):        8
# Vendor ID:           AuthenticAMD
# CPU family:          23
# Model:               49
# Model name:          AMD EPYC 7702 64-Core Processor
# Stepping:            0
# CPU MHz:             3353.158
# BogoMIPS:            3992.52
# Virtualization:      AMD-V
# L1d cache:           32K
# L1i cache:           32K
# L2 cache:            512K
# L3 cache:            16384K
# NUMA node0 CPU(s):   0-15
# NUMA node1 CPU(s):   16-31
# NUMA node2 CPU(s):   32-47
# NUMA node3 CPU(s):   48-63
# NUMA node4 CPU(s):   64-79
# NUMA node5 CPU(s):   80-95
# NUMA node6 CPU(s):   96-111
# NUMA node7 CPU(s):   112-127



# -------------------------------------------
# Single core run for each implementation
# OMP_NUM_THREADS=0
# -np 1
# -------------------------------------------
export NP_VALUE=1
export OMP_NUM_THREADS=
export STREAM_ARRAY_SIZE=2500000



#------------------------------------------------------------
# Setting vars for file paths to each STREAM implementation
#------------------------------------------------------------
export STREAM_DIR=$(pwd)
export OUTPUT_DIR=$STREAM_DIR/outputs
if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

export OUTPUT_FILE=$OUTPUT_DIR/single_core_output_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]]; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

echo "Running..."



#------------------------------------------------------------
# Load correct modules
#------------------------------------------------------------
module purge
module load gcc/10.2.0  intel-mpi/2019.10.317



#------------------------------------------------------------
# For all of the following make call you can add PFLAGS and CFLAGS.
# If you are running only on version of stream or the same flags between 
# the different versions of stream you can use export CFLAGS or export PFLAGS
# to define the variable.

# Use PFLAGS for program flags ie -DVERBOSE and -DDEBUG
# Use CFLAGS for compiler flags such as -fopenmp
#------------------------------------------------------------
make stream_original.exe        CFLAGS=""       PFLAGS=""
make stream_omp.exe             CFLAGS=""       PFLAGS=""
make stream_mpi_original.exe    CFLAGS=""       PFLAGS=""
make stream_mpi.exe             CFLAGS=""       PFLAGS=""
make stream_oshmem.exe          CFLAGS=""       PFLAGS=""


echo "=======================================================================================" >> $OUTPUT_FILE
echo "                               SINGLE CORE RUNS"                                         >> $OUTPUT_FILE
echo "=======================================================================================" >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
echo "         'Original' STREAM"           >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
if ./stream_original.exe >> $OUTPUT_FILE; then
        echo "Original impl finished."
else
        echo "FAILED TO RUN!" >> $OUTPUT_FILE
	echo "Original impl FAILED TO RUN!"
fi
echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE


echo "------------------------------------" >> $OUTPUT_FILE
echo "              OpenMP"                 >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
if ./stream_omp.exe >> $OUTPUT_FILE; then
        echo "OpenMP impl finished."
else
        echo "FAILED TO RUN!" >> $OUTPUT_FILE
        echo "OpenMP impl FAILED TO RUN!"
fi
echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE


echo "------------------------------------" >> $OUTPUT_FILE
echo "           'Original' MPI"            >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
if mpirun -n $NP_VALUE ./stream_mpi_original.exe >> $OUTPUT_FILE; then
        echo "Original MPI impl finished."
else
        echo "FAILED TO RUN!" >> $OUTPUT_FILE
	echo "Original MPI impl FAILED TO RUN!"
fi
echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE


echo "------------------------------------" >> $OUTPUT_FILE
echo "               MPI"                   >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
if mpirun -n $NP_VALUE ./stream_mpi.exe >> $OUTPUT_FILE; then
        echo "MPI impl finished."
else
        echo "FAILED TO RUN!" >> $OUTPUT_FILE
        echo "MPI impl FAILED TO RUN!"
fi
echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE


echo "------------------------------------" >> $OUTPUT_FILE
echo "             OpenSHMEM"               >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
if oshrun -n $NP_VALUE ./stream_oshmem.exe >> $OUTPUT_FILE; then
        echo "OpenSHMEM impl finished."
else
        echo "FAILED TO RUN!" >> $OUTPUT_FILE
        echo "OpenSHMEM impl FAILED TO RUN!"
fi
echo "Done! Output was directed to $OUTPUT_FILE"


make clean

echo "Would you like to see the results? (y/n)"
read RESPONSE
if [[ "${RESPONSE}" == "y" || "${RESPONSE}" == "Y" ]]; then
    cat $OUTPUT_FILE
    echo ""
    echo ""
fi
