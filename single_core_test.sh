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

#------------------------------------------------------------
# Setting vars for file paths to each STREAM implementation
#------------------------------------------------------------
export STREAM_ARRAY_SIZE=2500000
# export STREAM_ARRAY_SIZE=10000000

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

# -------------------------------------------
# Single core run for each implementation
# OMP_NUM_THREADS=0
# -np 1
# -------------------------------------------
export NP_VALUE=1

echo "=======================================================================================" >> $OUTPUT_FILE
echo "                               SINGLE CORE RUNS"                                         >> $OUTPUT_FILE
echo "=======================================================================================" >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
echo "         'Original' STREAM"           >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
gcc -DSTREAM_ARRAY_SIZE=$STREAM_ARRAY_SIZE $ORIGINAL_IMPL -o stream_original_serial
./stream_original_serial >> $OUTPUT_FILE
rm stream_original_serial

echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

echo "Original impl finished."

echo "------------------------------------" >> $OUTPUT_FILE
echo "              OpenMP"                 >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
gcc -DSTREAM_ARRAY_SIZE=$STREAM_ARRAY_SIZE $OMP_IMPL -o stream_openmp_serial
./stream_openmp_serial >> $OUTPUT_FILE
rm stream_openmp_serial

echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

echo "OpenMP impl finished."

echo "------------------------------------" >> $OUTPUT_FILE
echo "             OpenSHMEM"               >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
oshcc -DSTREAM_ARRAY_SIZE=$STREAM_ARRAY_SIZE $SHEM_IMPL -o stream_oshmem_serial
oshrun -np $NP_VALUE stream_oshmem_serial >> $OUTPUT_FILE
rm stream_oshmem_serial

echo >> $OUTPUT_FILE
echo >> $OUTPUT_FILE

echo "OpenSHMEM impl finished."

echo "------------------------------------" >> $OUTPUT_FILE
echo "               MPI"                   >> $OUTPUT_FILE
echo "------------------------------------" >> $OUTPUT_FILE
mpicc -DSTREAM_ARRAY_SIZE=$STREAM_ARRAY_SIZE $MPI_IMPL -o stream_mpi_serial
mpirun -np $NP_VALUE stream_mpi_serial >> $OUTPUT_FILE
rm stream_mpi_serial

echo "MPI impl finished."

echo "Done! Output was directed to $OUTPUT_FILE"





echo "Would you like to see the results? (y/n)"
export RESPONSE
read RESPONSE

if [[ "${RESPONSE}" == "y" || "${RESPONSE}" == "Y" ]]; then
    cat $OUTPUT_FILE
fi
