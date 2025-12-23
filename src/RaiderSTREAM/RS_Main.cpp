/**
 * @file RS_Main.cpp
 * @brief Main entry point for RaiderSTREAM benchmark
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights Reserved.
 * @author michael.beebe@ttu.edu
 * 
 * See LICENSE in the top level directory for licensing details
 */

#include <float.h>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

#ifdef _ENABLE_OMP_
#include "Impl/RS_OMP/RS_OMP.h"
#endif

#ifdef _ENABLE_OMP_TARGET_
#include "Impl/RS_OMP_TARGET/RS_OMP_TARGET.h"
#endif

#ifdef _ENABLE_OACC_
#include "Impl/RS_OACC/RS_OACC.h"
#endif

#ifdef _ENABLE_MPI_OMP_
#include "Impl/RS_MPI_OMP/RS_MPI_OMP.h"
#endif

#ifdef _ENABLE_SHMEM_OMP_
#include "Impl/RS_SHMEM_OMP/RS_SHMEM_OMP.h"
#endif

#ifdef _ENABLE_CUDA_
#include "Impl/RS_CUDA/RS_CUDA.cuh"
#endif

#ifdef _ENABLE_MPI_CUDA_
#include "Impl/RS_MPI_CUDA/RS_MPI_CUDA.cuh"
#endif

#ifdef _ENABLE_SHMEM_OMP_TARGET_
#include "Impl/RS_SHMEM_OMP_TARGET/RS_SHMEM_OMP_TARGET.h"
#endif

#ifdef _ENABLE_SHMEM_OACC_
#include "Impl/RS_SHMEM_OACC/RS_SHMEM_OACC.h"
#endif

#ifdef _ENABLE_SHMEM_CUDA_
#include "Impl/RS_SHMEM_CUDA/RS_SHMEM_CUDA.cuh"
#endif

/**
 * @brief Print timing results for a benchmark kernel
 * @param kernelName Name of the kernel being timed
 * @param totalRuntime Total runtime in seconds
 * @param MBPS Array of memory bandwidth results in MB/s
 * @param FLOPS Array of floating point operation rates in FLOP/s
 * @param kernelType Type of kernel being run
 * @param runKernelType Type of kernel requested to run
 * @param headerPrinted Whether the results header has been printed
 */
void printTiming(const std::string &kernelName, double totalRuntime,
                 const double *MBPS, const double *FLOPS,
                 RSBaseImpl::RSKernelType kernelType,
                 RSBaseImpl::RSKernelType runKernelType, bool &headerPrinted) {
  if (runKernelType == RSBaseImpl::RS_ALL || kernelType == runKernelType) {
    if (!headerPrinted) {
      std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
      std::cout << std::setfill(' ');
      std::cout << std::left << std::setw(30) << "Benchmark Kernel";
      std::cout << std::right << std::setw(20) << "Total Runtime (s)";
      std::cout << std::right << std::setw(20) << "MB/s";
      std::cout << std::right << std::setw(20) << "FLOP/s";
      std::cout << std::endl;
      std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
      std::cout << std::setfill(' ');
      headerPrinted = true;
    }

    if (kernelName.find("Copy") != std::string::npos) {
      std::cout << std::left << std::setw(30) << kernelName;
      std::cout << std::right << std::setw(20) << std::fixed
                << std::setprecision(6) << totalRuntime;
      std::cout << std::right << std::setw(20) << std::fixed
                << std::setprecision(0) << MBPS[kernelType];
      std::cout << std::right << std::setw(20) << std::fixed
                << std::setprecision(0) << "-";
      std::cout << std::endl;
    } else if (kernelName != "All") {
      std::cout << std::left << std::setw(30) << kernelName;
      std::cout << std::right << std::setw(20) << std::fixed
                << std::setprecision(6) << totalRuntime;
      std::cout << std::right << std::setw(20) << std::fixed
                << std::setprecision(0) << MBPS[kernelType];
      std::cout << std::right << std::setw(20) << std::fixed
                << std::setprecision(0) << FLOPS[kernelType];
      std::cout << std::endl;
    }
  }
}

#define PRINT_RUN_STAT(MESSAGE, VALUE)                                  \
  std::cout << std::left << std::setw(30) << MESSAGE;                   \
  if (VALUE != 0){                                                      \
    std::cout << std::right << std::setw(20) << std::fixed              \
              << std::setprecision(6) << VALUE << std::endl;            \
  } else {                                                              \
    std::cout << std::right << std::setw(20) << std::fixed              \
              << std::setprecision(6) << "-" << std::endl;              \
  }

void printRunStats(double SHMEM_MALLOC_TIME, double INIT_TIME, 
                 double RANDOM_GEN_TIME,double COLLECT_TIME){
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << std::setfill(' ');
  std::cout << std::left << std::setw(30) << "Operation";
  std::cout << std::right << std::setw(20) << "Time (s)" << std::endl;
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << std::setfill(' ');

  PRINT_RUN_STAT("Memmory alloc", SHMEM_MALLOC_TIME);
  PRINT_RUN_STAT("Array Initialization", INIT_TIME);
  PRINT_RUN_STAT("Random Array Gen", RANDOM_GEN_TIME);
  PRINT_RUN_STAT("Collect Time", COLLECT_TIME);
}

#ifdef _ENABLE_OMP_
/**
 * @brief Run OpenMP version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchOMP(RSOpts *Opts) {
  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_OMP object */
  RS_OMP *RS = new RS_OMP(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP OBJECT" << std::endl;
    return;
  }

  /* Initialize the RSRes object */
  RSRes *Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RSRes OBJECT" << std::endl;
  }

  /* Allocate Data */
  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, 
                  &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP" << std::endl;
    delete Results;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP" << std::endl;
    delete Results;
    RS->freeData();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP" << std::endl;
    delete Results;
    delete RS;
    return;
  }

  RS->collectChunks(&Results->COLLECT_TIME);

  /* Print the timing */
  Opts->printLogo();

  Opts->printOpts();
#pragma omp parallel
  {
#pragma omp single
    {
      std::cout << "RUNNING WITH NUM_THREADS = " << omp_get_num_threads()
                << std::endl;
    }
  }
  printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
  }

  /* Free the RS_OMP object */
  delete Results;
  delete RS;
}
#endif

#ifdef _ENABLE_OACC_
/**
 * @brief Run OpenACC version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchOACC(RSOpts *Opts) {

  /* Initialize the RS_OACC object */
  RS_OACC *RS = new RS_OACC(*Opts);
  if (!RS) {
    std::cout << "ERROR" << std::endl;
    return;
  }

  /* Initialize the RS_CUDA object */
  RSRes * Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RESULTS OBJECT" << std::endl;
	delete RS;
    return;
  }

  /* Set Device */
  if (!RS->setDevice()) {
    std::cout << "ERROR: COULD NOT SET DEVICE FOR RS_OACC" << std::endl;
    RS->freeData();
    acc_shutdown(acc_device_nvidia);
	delete Results;	
    delete RS;
    return;
  }
  /* Allocate Data */
  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OACC" << std::endl;
    RS->freeData();
    acc_shutdown(acc_device_nvidia);
	delete Results;
    delete RS;
    return;
  }

  /* Execute the Benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OACC" << std::endl;
    RS->freeData();
    acc_shutdown(acc_device_nvidia);
    delete Results;
	delete RS;
    return;
  }

  RS->collectChunks(&Results->COLLECT_TIME);

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << " ERROR: COULD NOT FREE THE MEMORY FOR RS_OACC" << std::endl;
    acc_shutdown(acc_device_nvidia);
	delete Results;
    delete RS;
    return;
  }

  /* Print the timing */
  Opts->printLogo();
  Opts->printOpts();
  printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
  }

  /* Free the RS_OACC and RSRes object */
  delete Results;
  delete RS;
}
#endif

#ifdef _ENABLE_OMP_TARGET_
/**
 * @brief Run OpenMP target offload version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchOMPTarget(RSOpts *Opts) {
  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_OMP object */
  RS_OMP_TARGET *RS = new RS_OMP_TARGET(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP_TARGET OBJECT" << std::endl;
    return;
  }

  /* Initialize the RSRes object */
  RSRes *Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RSRes OBJECT" << std::endl;
  }

  /* Allocate Data */
  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP_TARGET"
              << std::endl;
    delete Results;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP_TARGET"
              << std::endl;
    RS->freeData();
    delete Results;
    delete RS;
    return;
  }

  RS->collectChunks(&Results->COLLECT_TIME);

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP_TARGET"
              << std::endl;
    delete Results;
    delete RS;
    return;
  }

  /* Print the timing */
  Opts->printLogo();

  Opts->printOpts();
#pragma omp parallel
  {
#pragma omp single
    {
      std::cout << "RUNNING WITH NUM_THREADS = " << omp_get_num_threads()
                << std::endl;
    }
  }
  printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
  }

  /* Free the RS_OMP_TARGET and RSRes object */
  delete Results;
  delete RS;
}
#endif

#ifdef _ENABLE_MPI_OMP_
/**
 * @brief Run hybrid MPI+OpenMP version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchMPIOMP(RSOpts *Opts) {
  /* Initialize MPI */
  MPI_Init(NULL, NULL);
  int myRank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_MPI_OMP object */
  RS_MPI_OMP *RS = new RS_MPI_OMP(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_MPI_OMP OBJECT" << std::endl;
  }

  /* Initialize the RSRes object */
  RSRes *Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RSRes OBJECT" << std::endl;
  }

  /* Allocate Data */
  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_MPI_OMP" << std::endl;
    MPI_Finalize();
    delete Results;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_MPI_OMP"
              << std::endl;
    delete Results;
    RS->freeData();
    MPI_Finalize();
    delete RS;
    return;
  }
  
  /* Collect result data */
  RS->collectChunks(&Results->COLLECT_TIME);

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_MPI_OMP" << std::endl;
    delete Results;
    MPI_Finalize();
    delete RS;
    return;
  }

  /* Benchmark output */
  if (myRank == 0) {
    Opts->printLogo();
    Opts->printOpts();
#pragma omp parallel
    {
#pragma omp single
      {
        std::cout << "RUNNING WITH NUM_THREADS = " << omp_get_num_threads()
                  << std::endl;
      }
    }
    printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
    RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
    bool headerPrinted = false;
    for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
      RSBaseImpl::RSKernelType kernelType =
          static_cast<RSBaseImpl::RSKernelType>(i);
      std::string kernelName = BenchTypeTable[i].Notes;
      printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
    }
  }

  /* Free the RS_MPI_OMP and Results object, finalize MPI */
  delete Results;
  MPI_Finalize();
  delete RS;
}
#endif

#ifdef _ENABLE_SHMEM_OMP_
/**
 * @brief Run hybrid OpenSHMEM+OpenMP version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchSHMEMOMP(RSOpts *Opts) {
  /* Initialize OpenSHMEM */
  shmem_init();
  int myRank = shmem_my_pe();

  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_SHMEM_OMP object */
  RS_SHMEM_OMP *RS = new RS_SHMEM_OMP(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_SHMEM_OMP OBJECT" << std::endl;
  }
  
  /* Initialize the RSRes object */
  RSRes *Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RSRes OBJECT" << std::endl;
  }

  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_OMP"
              << std::endl;
    shmem_finalize();
    delete RS;
    delete Results;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_OMP"
              << std::endl;
    delete RS;
    RS->freeData();
    delete Results;
    shmem_finalize();
    return;
  }

  RS->collectChunks(&Results->COLLECT_TIME);

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_OMP"
              << std::endl;
    delete RS;
    delete Results;
    shmem_finalize();
    return;
  }


  /* Benchmark output */
  if (myRank == 0) {
    Opts->printLogo();
    Opts->printOpts();
#pragma omp parallel
    {
#pragma omp single
      {
        std::cout << "RUNNING WITH NUM_THREADS = " << omp_get_num_threads()
                  << std::endl;
      }
    }
    printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
    RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
    bool headerPrinted = false;
    for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
      RSBaseImpl::RSKernelType kernelType =
          static_cast<RSBaseImpl::RSKernelType>(i);
      std::string kernelName = BenchTypeTable[i].Notes;
      printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
    }
  }

  shmem_barrier_all();
  
  /* Free the RS_SHMEM_OMP object, finalize OpenSHMEM */
  delete Results;
  shmem_finalize();
  delete RS;
}
#endif

#ifdef _ENABLE_CUDA_
/**
 * @brief Run CUDA version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchCUDA(RSOpts *Opts) {
  /* Initialize the RS_CUDA object */
  RS_CUDA *RS = new RS_CUDA(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP OBJECT" << std::endl;
    return;
  }


  /* Initialize the RS_CUDA object */
  RSRes * Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RESULTS OBJECT" << std::endl;
	delete RS;
    return;
  }

  /* Allocate Data */
  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP" << std::endl;
    delete Results;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP" << std::endl;
    RS->freeData();
    delete Results;
    delete RS;
    return;
  }

  RS->collectChunks(&Results->COLLECT_TIME);

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP" << std::endl;
    delete Results;
    delete RS;
    return;
  }

  /* Print the timing */
  Opts->printLogo();
  Opts->printOpts();
  printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
  }

  /* Free the RS_OMP object */
  delete Results;
  delete RS;
}
#endif

#ifdef _ENABLE_MPI_CUDA_
/**
 * @brief Run hybrid MPI+CUDA version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 * @todo Implement MPI+CUDA version
 */
void runBenchMPICUDA(RSOpts *Opts) {
  // TODO: runBenchMPICUDA()
}
#endif

#ifdef _ENABLE_SHMEM_OMP_TARGET_
/**
 * @brief Run hybrid OpenSHMEM+OpenMP target offload version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchSHMEMOMPTARGET(RSOpts *Opts) {
  /* Initialize OpenSHMEM */
  shmem_init();
  int myRank = shmem_my_pe();
  

  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_SHMEM_OMP object */
  RS_SHMEM_OMP_TARGET *RS = new RS_SHMEM_OMP_TARGET(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_SHMEM_OMP OBJECT" << std::endl;
  }

  /* Initialize the RSRes object */
  RSRes *Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RSRes OBJECT" << std::endl;
		delete RS;
  }

  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_OMP_TARGET"
              << std::endl;
		delete Results;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_OMP_TARGET"
              << std::endl;
		delete Results;
    RS->freeData();
    shmem_finalize();
    delete RS;
    return;
  }
  
  RS->collectChunks(&Results->COLLECT_TIME);
  
  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_OMP_TARGET"
              << std::endl;
 		delete Results;
    shmem_finalize();
    delete RS;
    return;
  }


  /* Benchmark output */
  if (myRank == 0) {
    Opts->printLogo();
    Opts->printOpts();
// std::cout << "Symmetric heap size: " << shmem_info_get_heap_size() <<
// std::endl;
#pragma omp parallel
    {
#pragma omp single
      {
        std::cout << "RUNNING WITH NUM_THREADS = " << omp_get_num_threads()
                  << std::endl;
      }
    }
    printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
    RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
    bool headerPrinted = false;
    for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
      RSBaseImpl::RSKernelType kernelType =
          static_cast<RSBaseImpl::RSKernelType>(i);
      std::string kernelName = BenchTypeTable[i].Notes;
      printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
    }
  }

  shmem_barrier_all();

  /* Free the RS_SHMEM_OMP object, finalize OpenSHMEM */
	delete Results;
  shmem_finalize();
  delete RS;
}
#endif

#ifdef _ENABLE_SHMEM_OACC_
/**
 * @brief Run hybrid OpenSHMEM+OpenACC version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchSHMEMOACC(RSOpts *Opts) {
  /* Initialize OpenSHMEM */
  std::cout << _OPENACC <<std::endl;
  shmem_init();
  int myRank = shmem_my_pe();
  /* Initialize the RS_SHMEM_OMP object */
  RS_SHMEM_OACC *RS = new RS_SHMEM_OACC(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_SHMEM_OACC OBJECT" << std::endl;
  }
  /* Allocate Data */
  double *SHMEM_TIMES =
      static_cast<double *>(shmem_malloc(NUM_KERNELS * sizeof(double)));
  double *SHMEM_MBPS =
      static_cast<double *>(shmem_malloc(NUM_KERNELS * sizeof(double)));
  double *SHMEM_FLOPS =
      static_cast<double *>(shmem_malloc(NUM_KERNELS * sizeof(double)));

  double *SHMEM_BYTES =
      static_cast<double *>(shmem_malloc(NUM_KERNELS * sizeof(double)));
  double *SHMEM_FLOATOPS =
      static_cast<double *>(shmem_malloc(NUM_KERNELS * sizeof(double)));
  for (int i = 0; i < NUM_KERNELS; i++) {
    SHMEM_BYTES[i] = Opts->BYTES[i];
    SHMEM_FLOATOPS[i] = Opts->FLOATOPS[i];
  }

  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_OACC"
              << std::endl;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(SHMEM_TIMES, SHMEM_MBPS, SHMEM_FLOPS, SHMEM_BYTES,
                   SHMEM_FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_OACC"
              << std::endl;
    RS->freeData();
    shmem_finalize();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_OACC"
              << std::endl;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Benchmark output */
  if (myRank == 0) {
    Opts->printLogo();
    Opts->printOpts();
    RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
    bool headerPrinted = false;
    for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
      RSBaseImpl::RSKernelType kernelType =
          static_cast<RSBaseImpl::RSKernelType>(i);
      std::string kernelName = BenchTypeTable[i].Notes;
      printTiming(kernelName, SHMEM_TIMES[i], SHMEM_MBPS, SHMEM_FLOPS,
                  kernelType, runKernelType, headerPrinted);
    }
  }

  shmem_barrier_all();

  /* Free the RS_SHMEM_OMP object, finalize OpenSHMEM */
  shmem_free(SHMEM_TIMES);
  shmem_free(SHMEM_MBPS);
  shmem_free(SHMEM_FLOPS);
  shmem_free(SHMEM_BYTES);
  shmem_free(SHMEM_FLOATOPS);

  shmem_finalize();
  delete RS;
}
#endif

#ifdef _ENABLE_SHMEM_CUDA_
/**
 * @brief Run hybrid OpenSHMEM+CUDA version of benchmark
 * @param Opts Pointer to options object containing benchmark parameters
 */
void runBenchSHMEMCUDA(RSOpts *Opts) {
  /* Initialize SHMEM */
  shmem_init();
  int myRank = shmem_my_pe();

  /* Initialize the RS_SHMEM_CUDA object */
  RS_SHMEM_CUDA *RS = new RS_SHMEM_CUDA(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_SHMEM_CUDA OBJECT" << std::endl;
  }

  /* Initialize the RSRes object */
  RSRes *Results = new RSRes();
  if (!Results) {
    std::cout << "ERROR: COULD NOT ALLOCATE RSRes OBJECT" << std::endl;
		delete RS;
  }

  if (!RS->allocateData(&Results->ALLOC_TIME, &Results->INIT_TIME, &Results->RANDOM_GEN_TIME)) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_CUDA"
              << std::endl;
		delete Results;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Results->TIMES, Results->MBPS, Results->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_CUDA"
              << std::endl;
    RS->freeData();
		delete Results;
    shmem_finalize();
    delete RS;
    return;
  }

  RS->collectChunks(&Results->COLLECT_TIME);

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_CUDA"
              << std::endl;
    shmem_finalize();
		delete Results;
    delete RS;
    return;
  }

  /* Benchmark output */
  if (myRank == 0) {
    Opts->printLogo();
    Opts->printOpts();
    printRunStats(Results->ALLOC_TIME, Results->INIT_TIME, Results->RANDOM_GEN_TIME, Results->COLLECT_TIME);
    RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
    bool headerPrinted = false;
    for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
      RSBaseImpl::RSKernelType kernelType =
          static_cast<RSBaseImpl::RSKernelType>(i);
      std::string kernelName = BenchTypeTable[i].Notes;
      printTiming(kernelName, Results->TIMES[i], Results->MBPS, Results->FLOPS,
                  kernelType, runKernelType, headerPrinted);
    }
  }

  shmem_barrier_all();
  
  /* Free the RS_SHMEM_OMP object, finalize OpenSHMEM */
	delete Results;
  shmem_finalize();
  delete RS;
}
#endif

/**
 * @brief Main entry point for RaiderSTREAM benchmark
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 */
int main(int argc, char **argv) {
  RSOpts *Opts = new RSOpts();

  if (!Opts->parseOpts(argc, argv)) {
    std::cout << "Failed to parse command line options" << std::endl;
    delete Opts;
    return -1;
  }
#ifdef _ENABLE_OMP_
  runBenchOMP(Opts);
#endif

#ifdef _ENABLE_OMP_TARGET_
  runBenchOMPTarget(Opts);
#endif

#ifdef _ENABLE_OACC_
  runBenchOACC(Opts);
#endif

#ifdef _ENABLE_MPI_OMP_
  runBenchMPIOMP(Opts);
#endif

#ifdef _ENABLE_SHMEM_OMP_
  runBenchSHMEMOMP(Opts);
#endif

#ifdef _ENABLE_CUDA_
  runBenchCUDA(Opts);
#endif

#ifdef _ENABLE_MPI_CUDA_
  runBenchMPICUDA(Opts);
#endif

#ifdef _ENABLE_SHMEM_OMP_TARGET_
  runBenchSHMEMOMPTARGET(Opts);
#endif

#ifdef _ENABLE_SHMEM_CUDA_
  runBenchSHMEMCUDA(Opts);
#endif

#ifdef _ENABLE_SHMEM_OACC_
  runBenchSHMEMOACC(Opts);
#endif

  delete Opts;

  return 0;
}

/* EOF */
