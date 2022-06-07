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
# CPU MHz:             2663.352
# BogoMIPS:            3992.61
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
export STREAM_ARRAY_SIZE=2500000

#------------------------------------------------------------
# Setting vars for file paths to each STREAM implementation
#------------------------------------------------------------
export STREAM_DIR=$(pwd)
export OUTPUT_DIR=$STREAM_DIR/outputs
if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

export ORIGINAL_IMPL=$STREAM_DIR/stream_original.c
export OMP_IMPL=$STREAM_DIR/openmp/stream_openmp.c
export SHEM_IMPL=$STREAM_DIR/openshmem/stream_openshmem.c
export MPI_IMPL=$STREAM_DIR/mpi/stream_mpi.c


export OUTPUT_FILE=$OUTPUT_DIR/single_core_output_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]]; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

echo "Running..."


#------------------------------------------------------------
# Load necessary modules
#------------------------------------------------------------
module purge
module load gcc
module load openmpi
make all



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
echo "               MPI"                   >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
if mpirun -np $NP_VALUE stream_mpi.exe >> $OUTPUT_FILE; then
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
if oshrun -np $NP_VALUE stream_oshmem.exe >> $OUTPUT_FILE; then
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
