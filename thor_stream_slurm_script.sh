#!/bin/bash

# Architecture:        x86_64
# CPU op-mode(s):      32-bit, 64-bit
# Byte Order:          Little Endian
# CPU(s):              32
# On-line CPU(s) list: 0-31
# Thread(s) per core:  1
# Core(s) per socket:  16
# Socket(s):           2
# NUMA node(s):        2
# Vendor ID:            GenuineIntel
# CPU family:          6
# Model:               79
# Model name:          Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz
# Stepping:            1
# CPU MHz:             2599.948
# BogoMIPS:            5199.89
# Virtualization:      VT-x
# L1d cache:           32K
# L1i cache:           32K
# L2 cache:            256K
# L3 cache:            40960K
# NUMA node0 CPU(s):   0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
# NUMA node1 CPU(s):   1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31

# -------------------------------------------
# Slurm setup
# -------------------------------------------
#SBATCH -J STREAM_MULTI_TESTING
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p thor
#SBATCH --nodes=2
#SBATCH --sockets-per-node=2
#SBATCH --ntasks-per-node=32
#SBATCH -t 00:30:00

# -------------------------------------------
# Single core run for each implementation
# OMP_NUM_THREADS=0
# -np 1
# -------------------------------------------
export STREAM_ARRAY_SIZE=22000000
export OMP_NUM_THREADS=1   # 16 cores per socket
export NP_VALUE=2

#------------------------------------------------------------
# Setting vars for file paths to each STREAM implementation
#------------------------------------------------------------
export STREAM_DIR=$(pwd)
export OUTPUT_DIR=$STREAM_DIR/outputs_mc_o3_2n
if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

export OUTPUT_FILE=$OUTPUT_DIR/multi_node_output_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]]; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

#------------------------------------------------------------
# For all of the following make call you can add PFLAGS and CFLAGS.
# If you are running only on version of stream or the same flags between
# the different versions of stream you can use export CFLAGS or export PFLAGS
# to define the variable.

# Use PFLAGS for program flags ie -DVERBOSE and -DDEBUG
# Use CFLAGS for compiler flags such as -fopenmp
#------------------------------------------------------------
make stream_oshmem.exe          CFLAGS="-fopenmp -O3"       PFLAGS=""


i=1
for i in {1..5}
do
        echo "==========================================================================" >> $OUTPUT_FILE
        echo "    STREAM BENCHMARK RUN ON "$(date +"%d-%m-%y")" AT "$(date +"%T")         >> $OUTPUT_FILE
        echo "==========================================================================" >> $OUTPUT_FILE
        echo "-------------------------------------------------------------" >> $OUTPUT_FILE
        echo "                          OpenSHMEM"                           >> $OUTPUT_FILE
        echo "-------------------------------------------------------------" >> $OUTPUT_FILE
        if oshrun -np $NP_VALUE stream_oshmem.exe >> $OUTPUT_FILE; then
                echo "OpenSHMEM impl finished."
        else
                echo "FAILED TO RUN!" >> $OUTPUT_FILE
                echo "OpenSHMEM impl FAILED TO RUN!"
        fi

        echo "Done! Output was directed to $OUTPUT_FILE"
done


make clean