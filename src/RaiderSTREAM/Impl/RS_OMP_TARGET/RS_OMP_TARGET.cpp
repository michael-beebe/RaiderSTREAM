//
// _RS_OMP_TARGET_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_OMP_TARGET.h"

#ifdef _RS_OMP_TARGET_H_

/**************************************************
 * @brief Constructor for the RS_OMP_TARGET class.
 *
 * Initializes the RS_OMP_TARGET object with the specified options.
 *
 * @param opts Options for the RS_OMP_TARGET object.
 **************************************************/
RS_OMP_TARGET::RS_OMP_TARGET(const RSOpts &opts)
    : RSBaseImpl("RS_OMP_TARGET",
                 opts.getKernelTypeFromName(opts.getKernelName())),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()), numPEs(opts.getNumPEs()),
      numTeams(opts.getThreadBlocks()),
      threadsPerTeam(opts.getThreadsPerBlocks()), lArgc(0), lArgv(nullptr),
      d_a(nullptr), d_b(nullptr), d_c(nullptr), d_idx1(nullptr), d_idx2(nullptr),
      d_idx3(nullptr), scalar(3.0), device(omp_get_default_device()) {}

RS_OMP_TARGET::~RS_OMP_TARGET() {}

/**********************************************
 * @brief Allocates and initializes memory for
 *        data arrays and copies data to the device.
 *
 * @return True if allocation and copy are
 *         successful, false otherwise.
 **********************************************/
bool RS_OMP_TARGET::allocateData() {
  a = new STREAM_TYPE[streamArraySize];
  b = new STREAM_TYPE[streamArraySize];
  c = new STREAM_TYPE[streamArraySize];
  idx1 = new ssize_t[streamArraySize];
  idx2 = new ssize_t[streamArraySize];
  idx3 = new ssize_t[streamArraySize];

#ifdef _ARRAYGEN_
  initReadIdxArray(idx1, streamArraySize, "RaiderSTREAM/arraygen/IDX1.txt");
  initReadIdxArray(idx2, streamArraySize, "RaiderSTREAM/arraygen/IDX2.txt");
  initReadIdxArray(idx3, streamArraySize, "RaiderSTREAM/arraygen/IDX3.txt");
#else
  initRandomIdxArray(idx1, streamArraySize);
  initRandomIdxArray(idx2, streamArraySize);
  initRandomIdxArray(idx3, streamArraySize);
#endif

  size_t data_size = sizeof(double) * streamArraySize;
  size_t idx_size = sizeof(ssize_t) * streamArraySize;

  d_a = (double*) omp_target_alloc(data_size, device);
  d_b = (double*) omp_target_alloc(data_size, device);
  d_c = (double*) omp_target_alloc(data_size, device);
  d_idx1 = (ssize_t*) omp_target_alloc(idx_size, device);
  d_idx2 = (ssize_t*) omp_target_alloc(idx_size, device);
  d_idx3 = (ssize_t*) omp_target_alloc(idx_size, device);

  omp_target_memcpy(d_a, a, data_size, 0, 0, device, omp_get_initial_device());
  omp_target_memcpy(d_b, b, data_size, 0, 0, device, omp_get_initial_device());
  omp_target_memcpy(d_c, c, data_size, 0, 0, device, omp_get_initial_device());
  omp_target_memcpy(d_idx1, idx1, idx_size, 0, 0, device, omp_get_initial_device());
  omp_target_memcpy(d_idx2, idx2, idx_size, 0, 0, device, omp_get_initial_device());
  omp_target_memcpy(d_idx3, idx3, idx_size, 0, 0, device, omp_get_initial_device());

#ifdef _DEBUG_
  std::cout << "==============================================================="
               "===================="
            << std::endl;
  std::cout << " RaiderSTREAM Array Info:" << std::endl;
  std::cout << "==============================================================="
               "===================="
            << std::endl;
  std::cout << "streamArraySize         = " << streamArraySize << std::endl;
  std::cout << "a[streamArraySize-1]    = " << a[streamArraySize - 1]
            << std::endl;
  std::cout << "b[streamArraySize-1]    = " << b[streamArraySize - 1]
            << std::endl;
  std::cout << "c[streamArraySize-1]    = " << c[streamArraySize - 1]
            << std::endl;
  std::cout << "idx1[streamArraySize-1] = " << idx1[streamArraySize - 1]
            << std::endl;
  std::cout << "idx2[streamArraySize-1] = " << idx2[streamArraySize - 1]
            << std::endl;
  std::cout << "idx3[streamArraySize-1] = " << idx3[streamArraySize - 1]
            << std::endl;
  std::cout << "==============================================================="
               "===================="
            << std::endl;
#endif

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] idx1;
  delete[] idx2;
  delete[] idx3;

  return true;
}

/**************************************************
 * @brief Frees all allocated memory for the
 *        RS_OMP_TARGET object.
 *
 * This function deallocates memory for both host
 * and device pointers.
 *
 * @return true if all memory was successfully freed.
 **************************************************/
bool RS_OMP_TARGET::freeData() {
  if (d_a) {
    omp_target_free(d_a, device);
  }
  if (d_b) {
    omp_target_free(d_b, device);
  }
  if (d_c) {
    omp_target_free(d_c, device);
  }
  if (d_idx1) {
    omp_target_free(d_idx1, device);
  }
  if (d_idx2) {
    omp_target_free(d_idx2, device);
  }
  if (d_idx3) {
    omp_target_free(d_idx3, device);
  }
  return true;
}

/**************************************************
 * @brief Executes the specified kernel using OpenMP
 *        offloading.
 *
 * @param TIMES Array to store the execution times
 *              for each kernel.
 * @param MBPS Array to store the memory bandwidths
 *             for each kernel.
 * @param FLOPS Array to store the floating-point
 *              operation counts for each kernel.
 * @param BYTES Array to store the byte sizes for
 *              each kernel.
 * @param FLOATOPS Array to store the floating-point
 *                 operation sizes for each kernel.
 *
 * @return True if the execution was successful,
 *         false otherwise.
 **************************************************/
bool RS_OMP_TARGET::execute(double *TIMES, double *MBPS, double *FLOPS,
                            double *BYTES, double *FLOATOPS) {
  double start, end;
  double runTime = 0.0;
  double mbps = 0.0;
  double flops = 0.0;

  double* a = d_a;
  double* b = d_b;
  double* c = d_c;
  ssize_t* idx1 = d_idx1;
  ssize_t* idx2 = d_idx2;
  ssize_t* idx3 = d_idx3;

  RSBaseImpl::RSKernelType kType = getKernelType();

    seqCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
  switch (kType) {
  /* SEQUENTIAL KERNELS */
  case RSBaseImpl::RS_SEQ_COPY:
    start = mySecond();
    seqCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
    TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;
    break;

  case RSBaseImpl::RS_SEQ_SCALE:
    start = mySecond();
    seqScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;
    break;

  case RSBaseImpl::RS_SEQ_ADD:
    start = mySecond();
    seqAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;
    break;

  case RSBaseImpl::RS_SEQ_TRIAD:
    start = mySecond();
    seqTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;
    break;

  /* GATHER KERNELS */
  case RSBaseImpl::RS_GATHER_COPY:
    start = mySecond();
    gatherCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
    TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;
    break;

  case RSBaseImpl::RS_GATHER_SCALE:
    start = mySecond();
    gatherScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;
    break;

  case RSBaseImpl::RS_GATHER_ADD:
    start = mySecond();
    gatherAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;
    break;

  case RSBaseImpl::RS_GATHER_TRIAD:
    start = mySecond();
    gatherTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;
    break;

  /* SCATTER KERNELS */
  case RSBaseImpl::RS_SCATTER_COPY:
    start = mySecond();
    scatterCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;
    break;

  case RSBaseImpl::RS_SCATTER_SCALE:
    start = mySecond();
    scatterScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;
    break;

  case RSBaseImpl::RS_SCATTER_ADD:
    start = mySecond();
    scatterAdd(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;
    break;

  case RSBaseImpl::RS_SCATTER_TRIAD:
    start = mySecond();
    scatterTriad(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;
    break;

  /* SCATTER-GATHER KERNELS */
  case RSBaseImpl::RS_SG_COPY:
    start = mySecond();
    sgCopy(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
    TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SG_COPY] = flops;
    break;

  case RSBaseImpl::RS_SG_SCALE:
    start = mySecond();
    sgScale(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;
    break;

  case RSBaseImpl::RS_SG_ADD:
    start = mySecond();
    sgAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
    TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_ADD] = flops;
    break;

  case RSBaseImpl::RS_SG_TRIAD:
    start = mySecond();
    sgTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;
    break;

  /* CENTRAL KERNELS */
  case RSBaseImpl::RS_CENTRAL_COPY:
    start = mySecond();
    centralCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;
    break;

  case RSBaseImpl::RS_CENTRAL_SCALE:
    start = mySecond();
    centralScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;
    break;

  case RSBaseImpl::RS_CENTRAL_ADD:
    start = mySecond();
    centralAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;
    break;

  case RSBaseImpl::RS_CENTRAL_TRIAD:
    start = mySecond();
    centralTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;
    break;

  /* ALL KERNELS */
  case RSBaseImpl::RS_ALL:
    /* RS_SEQ_COPY */
    start = mySecond();
    seqCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
    TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;

    /* RS_SEQ_SCALE */
    start = mySecond();
    seqScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;

    /* RS_SEQ_ADD */
    start = mySecond();
    seqAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;

    /* RS_SEQ_TRIAD */
    start = mySecond();
    seqTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;

    /* RS_GATHER_COPY */
    start = mySecond();
    gatherCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
    TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;

    /* RS_GATHER_SCALE */
    start = mySecond();
    gatherScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;

    /* RS_GATHER_ADD */
    start = mySecond();
    gatherAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;

    /* RS_GATHER_TRIAD */
    start = mySecond();
    gatherTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;

    /* RS_SCATTER_COPY */
    start = mySecond();
    scatterCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;

    /* RS_SCATTER_SCALE */
    start = mySecond();
    scatterScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;

    /* RS_SCATTER_ADD */
    start = mySecond();
    scatterAdd(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;

    /* RS_SCATTER_TRIAD */
    start = mySecond();
    scatterTriad(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;

    /* RS_SG_COPY */
    start = mySecond();
    sgCopy(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
    TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SG_COPY] = flops;

    /* RS_SG_SCALE */
    start = mySecond();
    sgScale(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;

    /* RS_SG_ADD */
    start = mySecond();
    sgAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
    TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_ADD] = flops;

    /* RS_SG_TRIAD */
    start = mySecond();
    sgTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;

    /* RS_CENTRAL_COPY */
    start = mySecond();
    centralCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;

    /* RS_CENTRAL_SCALE */
    start = mySecond();
    centralScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;

    /* RS_CENTRAL_ADD */
    start = mySecond();
    centralAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;

    /* RS_CENTRAL_TRIAD */
    start = mySecond();
    centralTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
    end = mySecond();
    runTime = calculateRunTime(start, end);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;
    break;

  /* NO KERNELS, SOMETHING IS WRONG */
  default:
    std::cout << "RS_OMP_TARGET::execute() - ERROR: KERNEL NOT SET"
              << std::endl;
    return false;
  }
  return true;
}

#endif /* _RS_OMP_TARGET_H_ */
/* EOF */
