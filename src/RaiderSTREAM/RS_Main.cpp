//
// _RS_MAIN_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

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

void printTiming() {
  // TODO: printTiming()
}

#ifdef _ENABLE_OMP_
void runBenchOMP( RSOpts *Opts ) {
  // TODO: run_bench_omp()
}
#endif

#ifdef _ENABLE_MPI_OMP_
void runBenchMPIOMP( RSOpts *Opts ) {
  // TODO: runBenchMPIOMP()
}
#endif

#ifdef _ENABLE_OPENSHMEM_
void runBenchOpenSHMEM( RSOpts *Opts ) {
  // TODO: runBenchOpenSHMEM()
}
#endif

#ifdef _ENABLE_CUDA_
void runBenchCUDA( RSOpts *Opts ) {
  // TODO: runBenchCUDA()
}
#endif

#ifdef _ENABLE_MPI_CUDA_
void runBenchMPICUDA( RSOpts *Opts ) {
  // TODO: runBenchMPICUDA()
}
#endif

int main( int argc, char **argv ) {
  RSOpts *Opts = new RSOpts();
  
  if ( !Opts->parseOpts(argc, argv) ) {
    std::cout << "Failed to parse command line options" << std::endl;
    delete Opts;
    return -1;
  }
  
  #ifdef _ENABLE_OMP_
  runBenchOMP( Opts );
  #endif

  #ifdef _ENABLE_MPI_OMP_
  runBenchMPIOMP( Opts );
  #endif
  
  #ifdef _ENABLE_OPENSHMEM_
  runBenchOpenSHMEM( Opts );
  #endif
  
  #ifdef _ENABLE_CUDA_
  runBenchCUDA( Opts );
  #endif
  
  #ifdef _ENABLE_MPI_CUDA_
  runBenchMPICUDA( Opts );
  #endif

  #ifdef _DEBUG_
    printf("Hello, RaiderSTREAM!\n");
  #endif

  return 0;
}

// EOF