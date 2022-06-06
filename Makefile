STREAM_ARRAY_SIZE ?= 10000000
ORIGINAL_IMPL ?= stream_original.c
OMP_IMPL ?= stream_openmp.c
SHEM_IMPL ?= stream_openshmem.c
MPI_IMPL ?= stream_mpi.c

all: stream_original.exe stream_omp.exe stream_mpi.exe stream_oshmem.exe

stream_original.exe:
	gcc -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(ORIGINAL_IMPL) -o stream_original.exe

stream_omp.exe:
	gcc -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(OMP_IMPL) -o stream_omp.exe

stream_mpi.exe:
	mpicc -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(MPI_IMPL) -o stream_mpi.exe

stream_oshmem.exe:
	oshcc -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(SHEM_IMPL) -o stream_oshmem.exe

clean: 
	rm -f *.exe
