BUILD_DIR 	= ./build
SRC_DIR		= ./src
ROOT_DIR	= ./

ORIGINAL_IMPL 		?= $(SRC_DIR)/original/stream_original.c
OMP_IMPL 			?= $(SRC_DIR)/openmp/stream_openmp.c
MPI_IMPL 			?= $(SRC_DIR)/mpi/stream_mpi.c
SHEM_IMPL 			?= $(SRC_DIR)/openshmem/stream_openshmem.c
CUDA_IMPL			?= $(SRC_DIR)/cuda/stream_cuda.cu
CUDA_MPI_IMPL		?= $(SRC_DIR)/cuda-mpi/stream_cuda.cu

IDX1				?= $(ROOT_DIR)/IDX1.txt
IDX2				?= $(ROOT_DIR)/IDX2.txt
IDX3				?= $(ROOT_DIR)/IDX3.txt

ENABLE_OPENMP ?= false # Change this to false if you don't want to use OpenMP
ifeq ($(ENABLE_OPENMP), true)
OPENMP = -fopenmp
endif

ORIGINAL_FLAGS		?=
OMP_FLAGS			?=
MPI_FLAGS			?=
SHMEM_FLAGS			?=
CUDA_FLAGS			?=
CUDA_MPI_FLAGS		?= -DNUM_GPUS=2

#------------------------------------------------------------------
# 					 DO NOT EDIT BELOW
#------------------------------------------------------------------
all: build
	gcc $(ORIGINAL_FLAGS) $(OPENMP) $(ORIGINAL_IMPL) -o $(BUILD_DIR)/stream_original.exe
	gcc $(OMP_FLAGS) $(OPENMP) $(OMP_IMPL) -o $(BUILD_DIR)/stream_omp.exe
	mpicc $(MPI_FLAGS) $(OPENMP) $(MPI_IMPL) -o $(BUILD_DIR)/stream_mpi.exe
	oshcc $(SHMEM_FLAGS) $(OPENMP) $(SHEM_IMPL) -o $(BUILD_DIR)/stream_oshmem.exe
	nvcc $(CUDA_FLAGS) $(CUDA_IMPL) -o $(BUILD_DIR)/stream_cuda.exe
	nvcc -lmpi $(CUDA_MPI_FLAGS) $(CUDA_MPI_IMPL) -o $(BUILD_DIR)/stream_cuda_mpi.exe

stream_original: build
	gcc $(ORIGINAL_FLAGS) $(OPENMP) $(ORIGINAL_IMPL) -o $(BUILD_DIR)/stream_original.exe

stream_omp: build
	gcc $(OMP_FLAGS) $(OPENMP) $(OMP_IMPL) -o $(BUILD_DIR)/stream_omp.exe

stream_mpi: build
	mpicc $(MPI_FLAGS) $(OPENMP) $(MPI_IMPL) -o $(BUILD_DIR)/stream_mpi.exe

stream_oshmem: build
	oshcc $(SHMEM_FLAGS) $(OPENMP) $(SHEM_IMPL) -o $(BUILD_DIR)/stream_oshmem.exe

stream_cuda: build
	nvcc $(CUDA_FLAGS) $(CUDA_IMPL) -o $(BUILD_DIR)/stream_cuda.exe

stream_cuda_mpi: build
	nvcc -lmpi $(CUDA_MPI_FLAGS) $(CUDA_MPI_IMPL) -o $(BUILD_DIR)/stream_cuda_mpi.exe

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
	rm -rf $(IDX3)
	touch $(IDX1)
	touch $(IDX2)
	touch $(IDX3)

clean_all: clean clean_outputs clean_inputs

###############################################################################
