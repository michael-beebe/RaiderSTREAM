#!/bin/bash


# bugs		: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass
# bogomips	: 5788.84
# TLB size	: 3072 4K pages
# clflush size	: 64
# cache_alignment	: 64
# address sizes	: 43 bits physical, 48 bits virtual
# power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]
# processor	: 2
# vendor_id	: AuthenticAMD
# cpu family	: 23
# model		: 49
# model name	: AMD EPYC 7542 32-Core Processor
# stepping	: 0
# microcode	: 0x8301034
# cpu MHz		: 1794.551
# cache size	: 512 KB
# physical id	: 0
# siblings	: 64
# core id		: 2
# cpu cores	: 32
# apicid		: 4
# initial apicid	: 4
# fpu		: yes
# fpu_exception	: yes
# cpuid level	: 16
# wp		: yes
# -------------------------------
#     32 Cores per socket
#     2 Sockets per Node
# ------------------------------



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
make stream_original.exe CFLAGS="" PFLAGS="-DVERBOSE"
make stream_omp.exe CFLAGS="" PFLAGS="-DVERBOSE"
make stream_mpi.exe CFLAGS="" PFLAGS="-DVERBOSE"
make stream_oshmem.exe CFLAGS="" PFLAGS="-DVERBOSE"


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
