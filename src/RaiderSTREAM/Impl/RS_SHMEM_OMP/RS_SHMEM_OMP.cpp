//
// _RS_SHMEM_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_SHMEM_OMP.h"

#ifdef _RS_SHMEM_OMP_H_

#ifdef _DEBUG_
#define DBG(x)                                                                 \
  if (myRank == 0)                                                             \
    std::cout << "debug " #x " = " << x << std::endl;
#endif
#ifndef _DEBUG_
#define DBG(x)
#endif

#ifdef _SHMEM_1_5_
#define SHMEM_BENCHMARK(k, f)                                                  \
  do {                                                                         \
    shmem_barrier_all();                                                       \
    startTime = mySecond();                                                    \
    f;                                                                         \
    shmem_barrier_all();                                                       \
    endTime = mySecond();                                                      \
    runTime = calculateRunTime(startTime, endTime);                            \
    mbps = calculateMBPS(BYTES[k], runTime);                                   \
    flops = calculateFLOPS(FLOATOPS[k], runTime);                              \
    shmem_double_sum_reduce(SHMEM_TEAM_WORLD, totalRunTime, &runTime, 1);      \
    shmem_double_sum_reduce(SHMEM_TEAM_WORLD, totalMbps, &mbps, 1);            \
    shmem_double_sum_reduce(SHMEM_TEAM_WORLD, totalFlops, &flops, 1);          \
                                                                               \
    if (myRank == 0) {                                                         \
      TIMES[k] = *totalRunTime / size;                                         \
      MBPS[k] = *totalMbps / size;                                             \
      FLOPS[k] = *totalFlops / size;                                           \
    }                                                                          \
  } while (false)
#endif
#ifdef _SHMEM_1_4_
#define SHMEM_BENCHMARK(k, f)                                                  \
  do {                                                                         \
    shmem_barrier_all();                                                       \
    startTime = mySecond();                                                    \
    f;                                                                         \
    shmem_barrier_all();                                                       \
    endTime = mySecond();                                                      \
    runTime = calculateRunTime(startTime, endTime);                            \
    mbps = calculateMBPS(BYTES[k], runTime);                                   \
    flops = calculateFLOPS(FLOATOPS[k], runTime);                              \
    shmem_double_sum_to_all(totalRunTime, &runTime, 1, 0, 0, size, pWrk,       \
                            pSync);                                            \
    shmem_double_sum_to_all(totalMbps, &mbps, 1, 0, 0, size, pWrk, pSync);     \
    shmem_double_sum_to_all(totalFlops, &flops, 1, 0, 0, size, pWrk, pSync);   \
    DBG(runTime);                                                              \
    DBG(*totalRunTime);                                                        \
    DBG(TIMES[k]);                                                             \
    if (myRank == 0) {                                                         \
      TIMES[k] = *totalRunTime / size;                                         \
      MBPS[k] = *totalMbps / size;                                             \
      FLOPS[k] = *totalFlops / size;                                           \
    }                                                                          \
    DBG(TIMES[k]);                                                             \
  } while (false)
#endif

/**************************************************
 * @brief Constructor for the RS_SHMEM class.
 *
 * Initializes the RS_SHMEM object with the specified options.
 *
 * @param opts Options for the RS_SHMEM object.
 **************************************************/
RS_SHMEM_OMP::RS_SHMEM_OMP(const RSOpts &opts)
    : RSBaseImpl("RS_SHMEM_OMP",
                 opts.getKernelTypeFromName(opts.getKernelName())),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()), lArgc(0), lArgv(nullptr),
      numPEs(opts.getNumPEs()), a(nullptr), b(nullptr), idx1(nullptr),
      idx2(nullptr), idx3(nullptr), scalar(3.0) {}

RS_SHMEM_OMP::~RS_SHMEM_OMP() {}

/**********************************************
 * @brief Allocates and initializes memory
 *        for data arrays.
 *
 * @return True if allocation is
 *         successful, false otherwise.
 **********************************************/
bool RS_SHMEM_OMP::allocateData() {
  int myRank = shmem_my_pe(); /* Current rank */
  int size = shmem_n_pes();   /* Number of shmem ranks */

  if (numPEs == 0) {
    std::cout << "RS_SHMEM_OMP::allocateData() - ERROR: 'pes' cannot be 0"
              << std::endl;
    return false;
  }

  shmem_barrier_all();

  /* If updated, also update corresponding
   * region in RS_SHMEM_OMP::execute. */
  /* Calculate the chunk size for each rank */
  ssize_t chunkSize = streamArraySize / size;
  ssize_t remainder = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if (myRank == size - 1) {
    chunkSize += remainder;
  }

  /* Allocate memory for the local chunks in symmetric heap space */
  a = static_cast<double *>(shmem_malloc(chunkSize * sizeof(double)));
  b = static_cast<double *>(shmem_malloc(chunkSize * sizeof(double)));
  c = static_cast<double *>(shmem_malloc(chunkSize * sizeof(double)));
  idx1 = static_cast<ssize_t *>(shmem_malloc(chunkSize * sizeof(ssize_t)));
  idx2 = static_cast<ssize_t *>(shmem_malloc(chunkSize * sizeof(ssize_t)));
  idx3 = static_cast<ssize_t *>(shmem_malloc(chunkSize * sizeof(ssize_t)));

  /* Initialize the local chunks */
  initStreamArray(a, chunkSize, 1.0);
  initStreamArray(b, chunkSize, 2.0);
  initStreamArray(c, chunkSize, 0.0);

#ifdef _ARRAYGEN_
  initReadIdxArray(idx1, chunkSize, "RaiderSTREAM/arraygen/IDX1.txt");
  initReadIdxArray(idx2, chunkSize, "RaiderSTREAM/arraygen/IDX2.txt");
  initReadIdxArray(idx3, chunkSize, "RaiderSTREAM/arraygen/IDX3.txt");
#else
  initRandomIdxArray(idx1, chunkSize);
  initRandomIdxArray(idx2, chunkSize);
  initRandomIdxArray(idx3, chunkSize);
#endif

#ifdef _DEBUG_
  if (myRank == 0) {
    std::cout << "============================================================="
                 "======================"
              << std::endl;
    std::cout << " RaiderSTREAM Array Info:" << std::endl;
    std::cout << "============================================================="
                 "======================"
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
    std::cout << "============================================================="
                 "======================"
              << std::endl;
  }
#endif

  shmem_barrier_all();

  return true;
}

/**************************************************
 * @brief Frees all allocated memory for the
 *        RS_SHMEM object.
 *
 * @return true if all memory was successfully freed.
 **************************************************/
bool RS_SHMEM_OMP::freeData() {
  if (a) {
    shmem_free(a);
  }
  if (b) {
    shmem_free(b);
  }
  if (c) {
    shmem_free(c);
  }
  if (idx1) {
    shmem_free(idx1);
  }
  if (idx2) {
    shmem_free(idx2);
  }
  if (idx3) {
    shmem_free(idx3);
  }
  return true;
}

/**************************************************
 * @brief Executes the specified kernel using OpenSHMEM.
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
bool RS_SHMEM_OMP::execute(double *TIMES, double *MBPS, double *FLOPS,
                           double *BYTES, double *FLOATOPS) {
  double startTime = 0.0;
  double endTime = 0.0;
  double runTime = 0.0;
  double mbps = 0.0;
  double flops = 0.0;
  double *totalRunTime = (double *)shmem_malloc(sizeof(double));
  double *totalMbps = (double *)shmem_malloc(sizeof(double));
  double *totalFlops = (double *)shmem_malloc(sizeof(double));

  int myRank = shmem_my_pe(); /* Current rank */
  int size = shmem_n_pes();   /* Number of shmem ranks */
  size_t syncSize = SHMEM_SYNC_SIZE;

#ifdef _SHMEM_1_4_
  double *pWrk = static_cast<double *>(shmem_malloc(size * sizeof(double)));
  long *pSync = static_cast<long *>(shmem_malloc(syncSize * sizeof(long)));
  for (size_t i = 0; i < syncSize; ++i) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }
#endif

  shmem_barrier_all();

  /* If updated, also update corresponding
   * region in RS_SHMEM_OMP::allocateData. */
  /* Calculate the chunk size for each rank */
  ssize_t chunkSize = streamArraySize / size;
  ssize_t remainder = streamArraySize % size;
  ssize_t start = myRank * chunkSize;

  /* Adjust the chunk size for the last process */
  if (myRank == size - 1) {
    chunkSize += remainder;
  }

  RSBaseImpl::RSKernelType kType = getKernelType();

  switch (kType) {
  /* SEQUENTIAL KERNELS */
  case RSBaseImpl::RS_SEQ_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_COPY,
                    seqCopy(a, b, c, chunkSize, start));
    break;

  case RSBaseImpl::RS_SEQ_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_SCALE,
                    seqScale(a, b, c, chunkSize, start, scalar));
    break;

  case RSBaseImpl::RS_SEQ_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_ADD, seqAdd(a, b, c, chunkSize, start));
    break;

  case RSBaseImpl::RS_SEQ_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_TRIAD,
                    seqTriad(a, b, c, chunkSize, start, scalar));
    break;

  /* GATHER KERNELS */
  case RSBaseImpl::RS_GATHER_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_COPY,
                    gatherCopy(a, b, c, idx1, chunkSize, start));
    break;

  case RSBaseImpl::RS_GATHER_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_SCALE,
                    gatherScale(a, b, c, idx1, chunkSize, start, scalar));
    break;

  case RSBaseImpl::RS_GATHER_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_ADD,
                    gatherAdd(a, b, c, idx1, idx2, chunkSize, start));
    break;

  case RSBaseImpl::RS_GATHER_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_TRIAD,
                    gatherTriad(a, b, c, idx1, idx2, chunkSize, start, scalar));
    break;

  /* SCATTER KERNELS */
  case RSBaseImpl::RS_SCATTER_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_COPY,
                    scatterCopy(a, b, c, idx1, chunkSize, start));
    break;

  case RSBaseImpl::RS_SCATTER_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_SCALE,
                    scatterScale(a, b, c, idx1, chunkSize, start, scalar));
    break;

  case RSBaseImpl::RS_SCATTER_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_ADD,
                    scatterAdd(a, b, c, idx1, chunkSize, start));
    break;

  case RSBaseImpl::RS_SCATTER_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_TRIAD,
                    scatterTriad(a, b, c, idx1, chunkSize, start, scalar));
    break;

  /* SCATTER-GATHER KERNELS */
  case RSBaseImpl::RS_SG_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_COPY,
                    sgCopy(a, b, c, idx1, idx2, chunkSize, start));
    break;

  case RSBaseImpl::RS_SG_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_SCALE,
                    sgScale(a, b, c, idx1, idx2, chunkSize, start, scalar));
    break;

  case RSBaseImpl::RS_SG_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_ADD,
                    sgAdd(a, b, c, idx1, idx2, idx3, chunkSize, start));
    break;

  case RSBaseImpl::RS_SG_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_TRIAD, sgTriad(a, b, c, idx1, idx2, idx3,
                                                     chunkSize, start, scalar));
    break;

  /* CENTRAL KERNELS */
  case RSBaseImpl::RS_CENTRAL_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_COPY,
                    centralCopy(a, b, c, chunkSize, start));
    break;

  case RSBaseImpl::RS_CENTRAL_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_SCALE,
                    centralScale(a, b, c, chunkSize, start, scalar));
    break;

  case RSBaseImpl::RS_CENTRAL_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_ADD,
                    centralAdd(a, b, c, chunkSize, start));
    break;

  case RSBaseImpl::RS_CENTRAL_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_TRIAD,
                    centralTriad(a, b, c, chunkSize, start, scalar));
    break;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /* ALL KERNELS */
  case RSBaseImpl::RS_ALL:
    /* RS_SEQ_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_COPY,
                    seqCopy(a, b, c, chunkSize, start));

    /* RS_SEQ_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_SCALE,
                    seqScale(a, b, c, chunkSize, start, scalar));

    /* RS_SEQ_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_ADD, seqAdd(a, b, c, chunkSize, start));

    /* RS_SEQ_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_TRIAD,
                    seqTriad(a, b, c, chunkSize, start, scalar));

    /* RS_GATHER_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_COPY,
                    gatherCopy(a, b, c, idx1, chunkSize, start));

    /* RS_GATHER_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_SCALE,
                    gatherScale(a, b, c, idx1, chunkSize, start, scalar));

    /* RS_GATHER_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_ADD,
                    gatherAdd(a, b, c, idx1, idx2, chunkSize, start));

    /* RS_GATHER_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_TRIAD,
                    gatherTriad(a, b, c, idx1, idx2, chunkSize, start, scalar));

    /* RS_SCATTER_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_COPY,
                    scatterCopy(a, b, c, idx1, chunkSize, start));

    /* RS_SCATTER_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_SCALE,
                    scatterScale(a, b, c, idx1, chunkSize, start, scalar));

    /* RS_SCATTER_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_ADD,
                    scatterAdd(a, b, c, idx1, chunkSize, start));

    /* RS_SCATTER_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_TRIAD,
                    scatterTriad(a, b, c, idx1, chunkSize, start, scalar));

    /* RS_SG_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_COPY,
                    sgCopy(a, b, c, idx1, idx2, chunkSize, start));

    /* RS_SG_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_SCALE,
                    sgScale(a, b, c, idx1, idx2, chunkSize, start, scalar));

    /* RS_SG_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_ADD,
                    sgAdd(a, b, c, idx1, idx2, idx3, chunkSize, start));

    /* RS_SG_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_TRIAD, sgTriad(a, b, c, idx1, idx2, idx3,
                                                     chunkSize, start, scalar));

    /* RS_CENTRAL_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_COPY,
                    centralCopy(a, b, c, chunkSize, start));

    /* RS_CENTRAL_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_SCALE,
                    centralScale(a, b, c, chunkSize, start, scalar));

    /* RS_CENTRAL_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_ADD,
                    centralAdd(a, b, c, chunkSize, start));

    /* RS_CENTRAL_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_TRIAD,
                    centralTriad(a, b, c, chunkSize, start, scalar));

    break;

  /* NO KERNELS, SOMETHING IS WRONG */
  default:
    if (myRank == 0) {
      std::cout << "RS_SHMEM_OMP::execute() - ERROR: KERNEL NOT SET"
                << std::endl;
    }
#ifdef _SHMEM_1_4_
    shmem_free(pWrk);
    shmem_free(pSync);
#endif
    return false;
  }

#ifdef _SHMEM_1_4_
  shmem_free(pWrk);
  shmem_free(pSync);
#endif
  return true;
}

#endif /* _RS_SHMEM_OMP_H_ */

/* EOF */
