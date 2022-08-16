#!/bin/bash

# NOTE: STREAM_ARRAY_SIZE is set in the Makefile or can be exported globally

# -------------------------------------------------
#  Set true only for implementations you want to run
# -------------------------------------------------
export RUN_ORIGINAL=true
export RUN_OMP=true
export RUN_MPI=true
export RUM_SHMEM=true

# Don't forget to set OMP_NUM_THREADS if you are using OpenMP
export OMP_NUM_THREADS=1

# Set the number of PEs/ranks if using MPI and/or OpenSHMEM implementations
export NP_VALUE=

# Set this to true if you want this script to recompile the executables
export COMPILE=true

# Set this to true if you want to clean the build directory after the run
export CLEAN=false

# Set this to true if you want to be prompted to cat your output file. Good for a single run, not so good if you're running several runs at once
export PROMPT_OUTPUT=true

# -------------------------------------------------
#   Setting up directory to dump benchmark output
# -------------------------------------------------
export STREAM_DIR=$(pwd)
export OUTPUT_DIR=$STREAM_DIR/outputs
if [[ ! -d $OUTPUT_DIR ]] ; then
    mkdir $OUTPUT_DIR
fi

export OUTPUT_FILE=$OUTPUT_DIR/raiderstream_output_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]] ; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

export BUILD_DIR=$STREAM_DIR/build

# -------------------------------------------------
#   Compile each desired implementation
# -------------------------------------------------
if [[ $COMPILE == true ]] ; then
    if [[ $RUN_ORIGINAL == true ]] ; then
        make stream_original
    fi

    if [[ $RUN_OMP == true ]] ; then
        make stream_omp
    fi

    if [[ $RUN_MPI == true ]] ; then
        make stream_mpi
    fi

    if [[ $RUM_SHMEM == true ]] ; then
        make stream_oshmem
    fi
fi

echo "==========================================================================" >> $OUTPUT_FILE
echo "      RaiderSTREAM Run On "$(date +"%d-%m-%y")" AT "$(date +"%T")           >> $OUTPUT_FILE
echo "==========================================================================" >> $OUTPUT_FILE
if [[ $RUN_ORIGINAL == true ]] ; then
    echo "------------------------------------" >> $OUTPUT_FILE
    echo "         'Original' STREAM"           >> $OUTPUT_FILE
    echo "------------------------------------" >> $OUTPUT_FILE
    if $BUILD_DIR/stream_original.exe >> $OUTPUT_FILE; then
        echo "Original implementation finished."
    else
        echo "Original implementation failed to run!" >> $OUTPUT_FILE
        echo "Original implementation failed to run!"
    fi
    echo >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
fi

if [[ $RUN_OMP == true ]] ; then
    echo "------------------------------------" >> $OUTPUT_FILE
    echo "              OpenMP"                 >> $OUTPUT_FILE
    echo "------------------------------------" >> $OUTPUT_FILE
    if $BUILD_DIR/stream_omp.exe >> $OUTPUT_FILE; then
        echo "OpenMP implementation finished."
    else
        echo "OpenMP implementation failed to run!" >> $OUTPUT_FILE
        echo "OpenMP implementation failed to run!"
    fi
    echo >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
fi

if [[ $RUN_MPI == true ]] ; then
    echo "------------------------------------" >> $OUTPUT_FILE
    echo "                MPI"                  >> $OUTPUT_FILE
    echo "------------------------------------" >> $OUTPUT_FILE
    if mpirun -np $NP_VALUE $BUILD_DIR/stream_mpi.exe >> $OUTPUT_FILE; then
        echo "MPI implementation finished."
    else
        echo "MPI implementation failed to run!" >> $OUTPUT_FILE
        echo "MPI implementation failed to run!"
    fi
    echo >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
fi

if [[ $RUN_SHMEM == true ]] ; then
    echo "------------------------------------" >> $OUTPUT_FILE
    echo "            OpenSHMEM"                >> $OUTPUT_FILE
    echo "------------------------------------" >> $OUTPUT_FILE
    if oshrun -np $NP_VALUE $BUILD_DIR/stream_oshmem.exe >> $OUTPUT_FILE; then
        echo "OpenSHMEM implementation finished."
    else
        echo "OpenSHMEM implementation failed to run!" >> $OUTPUT_FILE
        echo "OpenSHMEM implementation failed to run!"
    fi
    echo >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
fi


echo "Done! Output was directed to $OUTPUT_FILE"

if [[ $CLEAN == true ]] ; then
    make clean > /dev/null 2>&1
fi

if [[ $PROMPT_OUTPUT == true ]] ; then
    echo "Would you like to see the results? (y/n)"
    read RESPONSE
    if [[ $RESPONSE == "y" || $RESPONSE == "Y" ]] ; then
        cat $OUTPUT_FILE
        echo ""
        echo ""
    fi
fi
