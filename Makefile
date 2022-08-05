BUILD_DIR 	= ./build
SRC_DIR		= ./src
MAKE_DIR	= ./

ORIGINAL_IMPL 		?= $(SRC_DIR)/stream_original.c
OMP_IMPL 			?= $(SRC_DIR)/stream_openmp.c
MPI_IMPL 			?= $(SRC_DIR)/stream_mpi.c
SHEM_IMPL 			?= $(SRC_DIR)/stream_openshmem.c

ENABLE_OPENMP ?= true
ifeq ($(ENABLE_OPENMP), true) # Change this to false if you don't want to use OpenMP
OPENMP = -fopenmp
endif


STREAM_ARRAY_SIZE 	?= 10000000

PFLAGS 				?= # Program-specific flags
CFLAGS 				?= # C Compiler flags
MPI_FLAGS			?= # MPI-specific flags
SHMEM_FLAGS			?= # OpenSHMEM-specifc flags


#------------------------------------------------------------------
# 					 DO NOT EDIT BELOW
#------------------------------------------------------------------
all: build
	gcc   $(CFLAGS) $(PFLAGS) $(OPENMP) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(ORIGINAL_IMPL) -o $(BUILD_DIR)/stream_original.exe
	gcc   $(CFLAGS) $(PFLAGS) $(OPENMP) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(OMP_IMPL) -o $(BUILD_DIR)/stream_omp.exe
	mpicc $(CFLAGS) $(PFLAGS) $(OPENMP) $(MPI_FLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(MPI_IMPL) -o $(BUILD_DIR)/stream_mpi.exe
	oshcc $(CFLAGS) $(PFLAGS) $(OPENMP) $(SHMEM_FLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(SHEM_IMPL) -o $(BUILD_DIR)/stream_oshmem.exe	

stream_original: build
	gcc $(CFLAGS) $(PFLAGS) $(OPENMP) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(ORIGINAL_OMP_IMPL) -o $(BUILD_DIR)/stream_original.exe

stream_omp: build
	gcc $(CFLAGS) $(PFLAGS) $(OPENMP) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(OMP_IMPL) -o $(BUILD_DIR)/stream_omp.exe

stream_mpi: build
	mpicc $(CFLAGS) $(PFLAGS) $(OPENMP) $(MPI_FLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(MPI_IMPL) -o $(BUILD_DIR)/stream_mpi.exe

stream_oshmem: build
	oshcc $(CFLAGS) $(PFLAGS) $(OPENMP) $(SHMEM_FLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(SHEM_IMPL) -o $(BUILD_DIR)/stream_oshmem.exe

build:
	@mkdir $(BUILD_DIR)

clean: 
	rm -f *.exe
	rm -rf $(BUILD_DIR)
