# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <time.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

#ifdef _ENABLE_OMP_
#include "Impl/RS_OMP/RS_OMP.h"
#endif

#ifdef _ENABLE_MPI_OMP_
#include "Impl/RS_MPI_OMP/RS_MPI_OMP.h"
#endif

#ifdef _ENABLE_OPENSHMEM_
#include "Impl/RS_OPENSHMEM/RS_SHMEM.h"
#endif

#ifdef _ENABLE_CUDA_
#include "Impl/RS_CUDA/RS_CUDA.cuh"
#endif

#ifdef _ENABLE_MPI_CUDA_
#include "Impl/RS_MPI_CUDA/RS_MPI_CUDA.cuh"
#endif

void print_timing() {
  // TODO: print_timing()
}


#ifdef _ENABLE_OMP_
void run_bench_omp( RSOpts *Opts ) {
  // TODO: run_bench_omp()
}
#endif

#ifdef _ENABLE_MPI_OMP_
void run_bench_mpi_omp( RSOpts *Opts ) {
  // TODO: run_bench_mpi_omp()
}
#endif

#ifdef _ENABLE_OPENSHMEM_
void run_bench_openshmem( RSOpts *Opts ) {
  // TODO: run_bench_openshmem()
}
#endif

#ifdef _ENABLE_CUDA_
void run_bench_cuda( RSOpts *Opts ) {
  // TODO: run_bench_cuda()
}
#endif

#ifdef _ENABLE_MPI_CUDA_
void run_bench_mpi_cuda( RSOpts *Opts ) {
  // TODO: run_bench_mpi_cuda()
}
#endif







int main( int argc, char **argv ) {
  RSOpts *Opts = new RSOpts();
  
  if ( !Opts->parse_opts(argc, argv) ) {
    std::cout << "Failed to parse command line options" << std::endl;
    delete Opts;
    return -1;
  }
  
  #ifdef _ENABLE_OMP_
  run_bench_omp( Opts );
  #endif

  #ifdef _ENABLE_MPI_OMP_
  run_bench_mpi_omp( Opts );
  #endif
  
  #ifdef _ENABLE_OPENSHMEM_
  run_bench_openshmem( Opts );
  #endif
  
  #ifdef _ENABLE_CUDA_
  run_bench_cuda( Opts );
  #endif
  
  #ifdef _ENABLE_MPI_CUDA_
  run_bench_mpi_cuda( Opts );
  #endif
  
  return 0;
}

// EOF