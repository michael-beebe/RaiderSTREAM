#!/bin/bash

# -------------------------------------------------
#  Set true only for implementations you want to run
# -------------------------------------------------
export RUN_ORIGINAL=true
export RUN_OMP=true
export RUN_MPI=true
export RUM_SHMEM=true

# Set this to true if you want this script to recompile the executables
export COMPILE=true

# -------------------------------------------------
#   Setting up directory to dump benchmark output
# -------------------------------------------------
export STREAM_DIR=$(pwd)
export OUTPUT_DIR=$STREAM_DIR/outputs
if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

export OUTPUT_FILE=$OUTPUT_DIR/raiderstream_output_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]]; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

export BUILD_DIR=$STREAM_DIR/build

# -------------------------------------------------
#   Compile each desired implementation
# -------------------------------------------------
if [[$RUN_ORIGINAL]] ; then
    make stream_original
fi


echo "==========================================================================" >> $OUTPUT_FILE
echo "      RaiderSTREAM Run On "$(date +"%d-%m-%y")" AT "$(date +"%T")           >> $OUTPUT_FILE
echo "==========================================================================" >> $OUTPUT_FILE
if [[$RUN_ORIGINAL]] ; then
    if ./$BUILD_DIR/stream_original.exe >> $OUTPUT_FILE; then
        echo "Original implementation finished."
    else
        echo "Original implementation failed to run!" >> $OUTPUT_FILE
        echo "Original implementation failed to run!"
    fi
    echo >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
fi

