//
// _RS_MAIN_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

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

/************************************************************************************/
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

/************************************************************************************/
#ifdef _ENABLE_OMP_
void runBenchOMP(RSOpts *Opts) {
  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_OMP object */
  RS_OMP *RS = new RS_OMP(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP OBJECT" << std::endl;
    return;
  }

  /* Allocate Data */
  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP" << std::endl;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Opts->TIMES, Opts->MBPS, Opts->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP" << std::endl;
    RS->freeData();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP" << std::endl;
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
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Opts->TIMES[i], Opts->MBPS, Opts->FLOPS, kernelType,
                runKernelType, headerPrinted);
  }

  /* Free the RS_OMP object */
  delete RS;
}
#endif

/***********************************************************************************/
#ifdef _ENABLE_OACC_
void runBenchOACC(RSOpts *Opts) {

  /* Initialize the RS_OACC object */
  RS_OACC *RS = new RS_OACC(*Opts);
  if (!RS) {
    std::cout << "ERROR" << std::endl;
    return;
  }
  /* Set Device */
  if (!RS->setDevice()) {
    std::cout << "ERROR: COULD NOT SET DEVICE FOR RS_OACC" << std::endl;
    RS->freeData();
    acc_shutdown(acc_device_nvidia);
    delete RS;
    return;
  }
  /* Allocate Data */
  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OACC" << std::endl;
    RS->freeData();
    acc_shutdown(acc_device_nvidia);
    delete RS;
    return;
  }

  /* Execute the Benchmark */
  if (!RS->execute(Opts->TIMES, Opts->MBPS, Opts->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OACC" << std::endl;
    RS->freeData();
    acc_shutdown(acc_device_nvidia);
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << " ERROR: COULD NOT FREE THE MEMORY FOR RS_OACC" << std::endl;
    acc_shutdown(acc_device_nvidia);
    delete RS;
    return;
  }

  /* Print the timing */
  Opts->printLogo();
  Opts->printOpts();
  // #pragma acc parallel
  // {
  //   int gang_num = acc_get_gang_num();
  //   int gang_size = acc_get_gang_size();
  //   int worker_num = acc_get_worker_num();
  //   int vector_length = acc_get_vector_length();

  //   #pragma acc serial
  //   {
  //     std::cout << "Gang number: " << gang_num << std::endl;
  //     std::cout << "Gang size: " << gang_size << std::endl;
  //     std::cout << "Worker number: " << worker_num << std::endl;
  //     std::cout << "Vector length: " << vector_length << std::endl;
  //   }
  // }
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Opts->TIMES[i], Opts->MBPS, Opts->FLOPS, kernelType,
                runKernelType, headerPrinted);
  }

  /* Free the RS_OACC object */
  delete RS;
}
#endif
/************************************************************************************/
#ifdef _ENABLE_OMP_TARGET_
void runBenchOMPTarget(RSOpts *Opts) {
  /* Initialize OpenMP */
  omp_get_num_threads();

  /* Initialize the RS_OMP object */
  RS_OMP_TARGET *RS = new RS_OMP_TARGET(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP_TARGET OBJECT" << std::endl;
    return;
  }

  /* Allocate Data */
  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP_TARGET"
              << std::endl;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Opts->TIMES, Opts->MBPS, Opts->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP_TARGET"
              << std::endl;
    RS->freeData();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP_TARGET"
              << std::endl;
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
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Opts->TIMES[i], Opts->MBPS, Opts->FLOPS, kernelType,
                runKernelType, headerPrinted);
  }

  /* Free the RS_OMP object */
  delete RS;
}
#endif
/************************************************************************************/
#ifdef _ENABLE_MPI_OMP_
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

  /* Allocate Data */
  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_MPI_OMP" << std::endl;
    MPI_Finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Opts->TIMES, Opts->MBPS, Opts->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_MPI_OMP"
              << std::endl;
    RS->freeData();
    MPI_Finalize();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_MPI_OMP" << std::endl;
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
    RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
    bool headerPrinted = false;
    for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
      RSBaseImpl::RSKernelType kernelType =
          static_cast<RSBaseImpl::RSKernelType>(i);
      std::string kernelName = BenchTypeTable[i].Notes;
      printTiming(kernelName, Opts->TIMES[i], Opts->MBPS, Opts->FLOPS,
                  kernelType, runKernelType, headerPrinted);
    }
  }

  /* Free the RS_MPI_OMP object, finalize MPI */
  MPI_Finalize();
  delete RS;
}
#endif

/************************************************************************************/
#ifdef _ENABLE_SHMEM_OMP_
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
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_OMP"
              << std::endl;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(SHMEM_TIMES, SHMEM_MBPS, SHMEM_FLOPS, SHMEM_BYTES,
                   SHMEM_FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_OMP"
              << std::endl;
    RS->freeData();
    shmem_finalize();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_OMP"
              << std::endl;
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

/************************************************************************************/
#ifdef _ENABLE_CUDA_
void runBenchCUDA(RSOpts *Opts) {
  /* Initialize the RS_CUDA object */
  RS_CUDA *RS = new RS_CUDA(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_OMP OBJECT" << std::endl;
    return;
  }

  /* Allocate Data */
  if (!RS->allocateData()) {
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_OMP" << std::endl;
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(Opts->TIMES, Opts->MBPS, Opts->FLOPS, Opts->BYTES,
                   Opts->FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_OMP" << std::endl;
    RS->freeData();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_OMP" << std::endl;
    delete RS;
    return;
  }

  /* Print the timing */
  Opts->printLogo();
  Opts->printOpts();
  RSBaseImpl::RSKernelType runKernelType = Opts->getKernelType();
  bool headerPrinted = false;
  for (int i = 0; i <= RSBaseImpl::RS_ALL; i++) {
    RSBaseImpl::RSKernelType kernelType =
        static_cast<RSBaseImpl::RSKernelType>(i);
    std::string kernelName = BenchTypeTable[i].Notes;
    printTiming(kernelName, Opts->TIMES[i], Opts->MBPS, Opts->FLOPS, kernelType,
                runKernelType, headerPrinted);
  }

  /* Free the RS_OMP object */
  delete RS;
}
#endif

/************************************************************************************/
#ifdef _ENABLE_MPI_CUDA_
void runBenchMPICUDA(RSOpts *Opts) {
  // TODO: runBenchMPICUDA()
}
#endif

#ifdef _ENABLE_SHMEM_OMP_TARGET_
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
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_OMP_TARGET"
              << std::endl;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(SHMEM_TIMES, SHMEM_MBPS, SHMEM_FLOPS, SHMEM_BYTES,
                   SHMEM_FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_OMP_TARGET"
              << std::endl;
    RS->freeData();
    shmem_finalize();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_OMP_TARGET"
              << std::endl;
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

#ifdef _ENABLE_SHMEM_OACC_
void runBenchSHMEMOACC(RSOpts *Opts) {
  /* Initialize OpenSHMEM */
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
// std::cout << "Symmetric heap size: " << shmem_info_get_heap_size() <<
// std::endl;
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
void runBenchSHMEMCUDA(RSOpts *Opts) {
  /* Initialize SHMEM */
  shmem_init();
  int myRank = shmem_my_pe();

  /* Initialize the RS_SHMEM_OMP object */
  RS_SHMEM_CUDA *RS = new RS_SHMEM_CUDA(*Opts);
  if (!RS) {
    std::cout << "ERROR: COULD NOT ALLOCATE RS_SHMEM_CUDA OBJECT" << std::endl;
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
    std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR RS_SHMEM_CUDA"
              << std::endl;
    shmem_finalize();
    delete RS;
    return;
  }

  /* Execute the benchmark */
  if (!RS->execute(SHMEM_TIMES, SHMEM_MBPS, SHMEM_FLOPS, SHMEM_BYTES,
                   SHMEM_FLOATOPS)) {
    std::cout << "ERROR: COULD NOT EXECUTE BENCHMARK FOR RS_SHMEM_CUDA"
              << std::endl;
    RS->freeData();
    shmem_finalize();
    delete RS;
    return;
  }

  /* Free the data */
  if (!RS->freeData()) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR RS_SHMEM_CUDA"
              << std::endl;
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
/************************************************************************************/
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

  return 0;
}
/************************************************************************************/

/* EOF */
