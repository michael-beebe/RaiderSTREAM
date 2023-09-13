#!/bin/bash
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