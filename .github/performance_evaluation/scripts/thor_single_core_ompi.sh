#!/bin/bash
# -------------------------------------------
# Single core run for each implementation
# OMP_NUM_THREADS=0
# -np 1
# -------------------------------------------
export NP_VALUE=1
export OMP_NUM_THREADS=
export STREAM_ARRAY_SIZE=22000000



#------------------------------------------------------------
# Setting vars for file paths to each STREAM implementation
#------------------------------------------------------------
export STREAM_DIR=$(pwd)
export OUTPUT_DIR=$STREAM_DIR/outputs_sc_o3
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
# For all of the following make call you can add PFLAGS and CFLAGS.
# If you are running only on version of stream or the same flags between 
# the different versions of stream you can use export CFLAGS or export PFLAGS
# to define the variable.

# Use PFLAGS for program flags ie -DVERBOSE and -DDEBUG
# Use CFLAGS for compiler flags such as -fopenmp
#------------------------------------------------------------
# make stream_original.exe        CFLAGS="-O3"       PFLAGS=""
# make stream_omp.exe             CFLAGS="-O3"       PFLAGS=""
# make stream_mpi_original.exe    CFLAGS="-O3"       PFLAGS=""
# make stream_mpi.exe             CFLAGS="-O3"       PFLAGS=""
make stream_oshmem.exe          CFLAGS="-O3"       PFLAGS=""


echo "=======================================================================================" >> $OUTPUT_FILE
echo "                               SINGLE CORE RUNS"                                         >> $OUTPUT_FILE
echo "=======================================================================================" >> $OUTPUT_FILE
# echo "------------------------------------" >> $OUTPUT_FILE
# echo "         'Original' STREAM"           >> $OUTPUT_FILE
# echo "------------------------------------" >> $OUTPUT_FILE
# if ./stream_original.exe >> $OUTPUT_FILE; then
#         echo "Original impl finished."
# else
#         echo "FAILED TO RUN!" >> $OUTPUT_FILE
# 	echo "Original impl FAILED TO RUN!"
# fi
# echo >> $OUTPUT_FILE
# echo >> $OUTPUT_FILE


# echo "------------------------------------" >> $OUTPUT_FILE
# echo "              OpenMP"                 >> $OUTPUT_FILE
# echo "------------------------------------" >> $OUTPUT_FILE
# if ./stream_omp.exe >> $OUTPUT_FILE; then
#         echo "OpenMP impl finished."
# else
#         echo "FAILED TO RUN!" >> $OUTPUT_FILE
#         echo "OpenMP impl FAILED TO RUN!"
# fi
# echo >> $OUTPUT_FILE
# echo >> $OUTPUT_FILE


# echo "------------------------------------" >> $OUTPUT_FILE
# echo "           'Original' MPI"           >> $OUTPUT_FILE
# echo "------------------------------------" >> $OUTPUT_FILE
# if mpirun -np $NP_VALUE stream_mpi_original.exe >> $OUTPUT_FILE; then
#         echo "Original MPI impl finished."
# else
#         echo "FAILED TO RUN!" >> $OUTPUT_FILE
# 	echo "Original MPI impl FAILED TO RUN!"
# fi
# echo >> $OUTPUT_FILE
# echo >> $OUTPUT_FILE


# echo "------------------------------------" >> $OUTPUT_FILE
# echo "               MPI"                   >> $OUTPUT_FILE
# echo "------------------------------------" >> $OUTPUT_FILE
# if mpirun -np $NP_VALUE stream_mpi.exe >> $OUTPUT_FILE; then
#         echo "MPI impl finished."
# else
#         echo "FAILED TO RUN!" >> $OUTPUT_FILE
#         echo "MPI impl FAILED TO RUN!"
# fi
# echo >> $OUTPUT_FILE
# echo >> $OUTPUT_FILE


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