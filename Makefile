STREAM_ARRAY_SIZE ?= 10000000
ORIGINAL_OMP_IMPL ?= ./stream_original.c
ORIGINAL_MPI_IMPL ?= ./mpi/stream_mpi_original.c
OMP_IMPL ?= ./openmp/stream_openmp.c
MPI_IMPL ?= ./mpi/stream_mpi.c
SHEM_IMPL ?= ./openshmem/stream_openshmem.c
CFLAGS ?=
PFLAGS ?= 

# STREAM_ARRAY_SIZE ?= 10000000
# ORIGINAL_OMP_IMPL ?= stream_original.c
# ORIGINAL_MPI_IMPL ?= mpi/original_stream_mpi.c
# OMP_IMPL ?= openmp/stream_openmp.c
# MPI_IMPL ?= mpi/stream_mpi.c
# SHEM_IMPL ?= openshmem/stream_openshmem.c
# CFLAGS ?=
# PFLAGS ?=

stream_original.exe:
	gcc $(CFLAGS) $(PFLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(ORIGINAL_OMP_IMPL) -o stream_original.exe

stream_mpi_original.exe:
	mpicc $(CFLAGS) $(PFLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(ORIGINAL_MPI_IMPL) -o stream_mpi_original.exe

stream_omp.exe:
	gcc $(CFLAGS) $(PFLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(OMP_IMPL) -o stream_omp.exe

stream_mpi.exe:
	mpicc $(CFLAGS) $(PFLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(MPI_IMPL) -o stream_mpi.exe

stream_oshmem.exe:
	oshcc $(CFLAGS) $(PFLAGS) -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) $(SHEM_IMPL) -o stream_oshmem.exe

clean: 
	rm -f *.exe
