//
// _RS_MAIN_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <iomanip>
#include <string>

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

void printTiming(
  const std::string& kernelName,
  double totalRuntime, const double* MBPS, const double* FLOPS,
  RSBaseImpl::RSKernelType kernelType, RSBaseImpl::RSKernelType runKernelType,
  bool& headerPrinted
) {
  if (runKernelType == RSBaseImpl::RS_ALL || kernelType == runKernelType) {
    if (!headerPrinted) {
      std::cout << std::setfill('-') << std::setw(90) << "-" << std::endl;
      std::cout << std::setfill(' ');
      std::cout << std::left << std::setw(30) << "Benchmark Kernel";
      std::cout << std::right << std::setw(20) << "Total Runtime (s)";
      std::cout << std::right << std::setw(20) << "MB/s";
      std::cout << std::right << std::setw(20) << "FLOP/s";
      std::cout << std::endl;
      std::cout << std::setfill('-') << std::setw(90) << "-" << std::endl;
      std::cout << std::setfill(' ');
      headerPrinted = true;
    }

    if (kernelName.find("Copy") != std::string::npos) {
      std::cout << std::left << std::setw(30) << kernelName;
      std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(6) << totalRuntime;
      std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(0) << MBPS[kernelType];
      std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(0) << "-";
      std::cout << std::endl;
    }
    else if (kernelName != "All") {
      std::cout << std::left << std::setw(30) << kernelName;
      std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(6) << totalRuntime;
      std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(0) << MBPS[kernelType];
      std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(0) << FLOPS[kernelType];
      std::cout << std::endl;
    }
  }
}

#ifdef _ENABLE_OMP_
void runBenchOMP(RSOpts *Opts) {
  /* Initialize the RS_OMP object */
  RS_OMP *RS = new RS_OMP(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP OBJECT" << std::endl;
    return;
  }

  #ifdef _DEBUG_
  Opts->printOpts();
  #endif

  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP" << std::endl;
    delete RS;
    return;
  }

  /* Execute the benchmark */ 
  if (!RS->execute(Opts->TIMES, Opts->MBPS, Opts->FLOPS, Opts->BYTES, Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP" << std::endl;
    RS->freeData();
    delete RS;
    return;
  }

  /* Print the timing */
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType = static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Opts->TIMES[i], Opts->MBPS, Opts->FLOPS, kernelType, runKernelType, headerPrinted);
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP" << std::endl;
    delete RS;
    return;
  }

  /* Free the RS_OMP object */
  delete RS;
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

  return 0;
}

// EOF