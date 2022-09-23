BUILD_DIR 	= ./build
SRC_DIR		= ./src
ROOT_DIR	= ./

ORIGINAL_IMPL 		?= $(SRC_DIR)/original/stream_original.c
OMP_IMPL 			?= $(SRC_DIR)/openmp/stream_openmp.c
MPI_IMPL 			?= $(SRC_DIR)/mpi/stream_mpi.c
SHEM_IMPL 			?= $(SRC_DIR)/openshmem/stream_openshmem.c
CUDA_IMPL			?= $(SRC_DIR)/cuda/stream_cuda.cu

IDX1				?= $(ROOT_DIR)/IDX1.txt
IDX2				?= $(ROOT_DIR)/IDX2.txt
IDX3				?= $(ROOT_DIR)/IDX3.txt

ENABLE_OPENMP ?= false # Change this to false if you don't want to use OpenMP
ifeq ($(ENABLE_OPENMP), true)
OPENMP = -fopenmp
endif

PFLAGS 				?= #-DCUSTOM # Program-specific flags
CFLAGS 				?= # C Compiler flags
MPI_FLAGS			?= # MPI-specific flags
SHMEM_FLAGS			?= # OpenSHMEM-specifc flags
CUDA_FLAGS			?= # CUDA-specific flags

#------------------------------------------------------------------
# 					 DO NOT EDIT BELOW
#------------------------------------------------------------------
all: build
	gcc   $(CFLAGS) $(PFLAGS) $(OPENMP) $(ORIGINAL_IMPL) 			 -o $(BUILD_DIR)/stream_original.exe
	gcc   $(CFLAGS) $(PFLAGS) $(OPENMP) $(OMP_IMPL) 				 -o $(BUILD_DIR)/stream_omp.exe
	mpicc $(CFLAGS) $(PFLAGS) $(OPENMP) $(MPI_FLAGS)  	$(MPI_IMPL)  -o $(BUILD_DIR)/stream_mpi.exe
	oshcc $(CFLAGS) $(PFLAGS) $(OPENMP) $(SHMEM_FLAGS)  $(SHEM_IMPL) -o $(BUILD_DIR)/stream_oshmem.exe
	nvcc  $(CFLAGS) $(PFLAGS) $(OPENMP) $(CUDA_FLAGS)	$(CUDA_IMPL) -o $(BUILD_DIR)/stream_cuda.exe

stream_original: build
	gcc $(CFLAGS) $(PFLAGS) $(OPENMP)  $(ORIGINAL_IMPL) -o $(BUILD_DIR)/stream_original.exe

stream_omp: build
	gcc $(CFLAGS) $(PFLAGS) $(OPENMP)  $(OMP_IMPL) -o $(BUILD_DIR)/stream_omp.exe

stream_mpi: build
	mpicc $(CFLAGS) $(PFLAGS) $(OPENMP) $(MPI_FLAGS)  $(MPI_IMPL) -o $(BUILD_DIR)/stream_mpi.exe

stream_oshmem: build
	oshcc $(CFLAGS) $(PFLAGS) $(OPENMP) $(SHMEM_FLAGS)  $(SHEM_IMPL) -o $(BUILD_DIR)/stream_oshmem.exe

stream_cuda: build
	nvcc  $(CFLAGS) $(PFLAGS) $(OPENMP) $(CUDA_FLAGS)	$(CUDA_IMPL) -o $(BUILD_DIR)/stream_cuda.exe

build:
	@mkdir $(BUILD_DIR)

clean:
	rm -f *.exe
	rm -rf $(BUILD_DIR)

clean_outputs:
	@rm -rf $(ROOT_DIR)/outputs

clean_inputs:
	rm -rf $(IDX1)
	rm -rf $(IDX2)
	touch $(IDX1)
	touch $(IDX2)

clean_all: clean clean_outputs clean_inputs

###############################################################################
