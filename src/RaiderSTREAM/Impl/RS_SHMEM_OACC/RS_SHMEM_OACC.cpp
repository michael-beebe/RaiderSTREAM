//
// _RS_SHMEM_OACC_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_SHMEM_OACC.h"

#ifdef _RS_SHMEM_OACC_H_

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
RS_SHMEM_OACC::RS_SHMEM_OACC(const RSOpts &opts)
    : RSBaseImpl("RS_SHMEM_OACC",
                 opts.getKernelTypeFromName(opts.getKernelName())),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()), lArgc(0), lArgv(nullptr),
      numPEs(opts.getNumPEs()), d_a(nullptr), d_b(nullptr), d_idx1(nullptr),
      d_idx2(nullptr), d_idx3(nullptr), scalar(3) {}

RS_SHMEM_OACC::~RS_SHMEM_OACC() {}

/**********************************************
 * @brief Allocates and initializes memory
 *        for data arrays.
 *
 * @return True if allocation is
 *         successful, false otherwise.
 **********************************************/
bool RS_SHMEM_OACC::allocateData() {
  int myRank = shmem_my_pe(); /* Current rank */
  int size = shmem_n_pes();   /* Number of shmem ranks */

  if (numPEs == 0) {
    std::cout << "RS_SHMEM_OACC::allocateData() - ERROR: 'pes' cannot be 0"
              << std::endl;
    return false;
  }

  shmem_barrier_all();

  /* If updated, also update corresponding
   * region in RS_SHMEM_OACC::execute. */
  /* Calculate the chunk size for each rank */
  ssize_t chunkSize = streamArraySize / size;
  ssize_t remainder = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if (myRank == size - 1) {
    chunkSize += remainder;
  }

  /* Allocate memory for the local chunks in symmetric heap space */
  STREAM_TYPE *a =
      static_cast<STREAM_TYPE *>(shmem_malloc(chunkSize * sizeof(STREAM_TYPE)));
  STREAM_TYPE *b =
      static_cast<STREAM_TYPE *>(shmem_malloc(chunkSize * sizeof(STREAM_TYPE)));
  STREAM_TYPE *c =
      static_cast<STREAM_TYPE *>(shmem_malloc(chunkSize * sizeof(STREAM_TYPE)));
  ssize_t *idx1 =
      static_cast<ssize_t *>(shmem_malloc(chunkSize * sizeof(ssize_t)));
  ssize_t *idx2 =
      static_cast<ssize_t *>(shmem_malloc(chunkSize * sizeof(ssize_t)));
  ssize_t *idx3 =
      static_cast<ssize_t *>(shmem_malloc(chunkSize * sizeof(ssize_t)));

  /* Initialize the local chunks */
  initStreamArray(a, chunkSize, 1);
  initStreamArray(b, chunkSize, 2);
  initStreamArray(c, chunkSize, 0);

#ifdef _ARRAYGEN_
  initReadIdxArray(idx1, chunkSize, "RaiderSTREAM/arraygen/IDX1.txt");
  initReadIdxArray(idx2, chunkSize, "RaiderSTREAM/arraygen/IDX2.txt");
  initReadIdxArray(idx3, chunkSize, "RaiderSTREAM/arraygen/IDX3.txt");
#else
  initRandomIdxArray(idx1, chunkSize);
  initRandomIdxArray(idx2, chunkSize);
  initRandomIdxArray(idx3, chunkSize);
#endif

  ssize_t streamMemArraySize = sizeof(STREAM_TYPE) * chunkSize;
  ssize_t idxMemArraySize = sizeof(ssize_t) * chunkSize;

  d_a = (STREAM_TYPE *)acc_malloc(streamMemArraySize);
  acc_memcpy_to_device(d_a, a, streamMemArraySize);

  /* b -> d_b */
  d_b = (STREAM_TYPE *)acc_malloc(streamMemArraySize);
  acc_memcpy_to_device(d_b, b, streamMemArraySize);

  /* c -> d_c */
  d_c = (STREAM_TYPE *)acc_malloc(streamMemArraySize);
  acc_memcpy_to_device(d_c, c, streamMemArraySize);

  /* idx1 -> d_idx1 */
  d_idx1 = (ssize_t *)acc_malloc(idxMemArraySize);
  acc_memcpy_to_device(d_idx1, idx1, idxMemArraySize);

  /* idx2 -> d_idx2 */
  d_idx2 = (ssize_t *)acc_malloc(idxMemArraySize);
  acc_memcpy_to_device(d_idx2, idx2, idxMemArraySize);

  /* idx3 -> d_idx3 */
  d_idx3 = (ssize_t *)acc_malloc(idxMemArraySize);
  acc_memcpy_to_device(d_idx3, idx3, idxMemArraySize);

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
 * @brief Frees all allocated memory for the
 *        RS_SHMEM object.
 *
 * @return true if all memory was successfully freed.
 **************************************************/
bool RS_SHMEM_OACC::freeData() {
  if (d_a) {
    acc_free(d_a);
  }
  if (d_b) {
    acc_free(d_b);
  }
  if (d_c) {
    acc_free(d_c);
  }
  if (d_idx1) {
    acc_free(d_idx1);
  }
  if (d_idx2) {
    acc_free(d_idx2);
  }
  if (d_idx3) {
    acc_free(d_idx3);
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
bool RS_SHMEM_OACC::execute(double *TIMES, double *MBPS, double *FLOPS,
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
   * region in RS_SHMEM_OACC::allocateData. */
  /* Calculate the chunk size for each rank */
  ssize_t chunkSize = streamArraySize / size;
  ssize_t remainder = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if (myRank == size - 1) {
    chunkSize += remainder;
  }

  RSBaseImpl::RSKernelType kType = getKernelType();

  switch (kType) {
  /* SEQUENTIAL KERNELS */
  case RSBaseImpl::RS_SEQ_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_COPY, seqCopy(d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_SEQ_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_SCALE,
                    seqScale(d_a, d_b, d_c, chunkSize, scalar));
    break;

  case RSBaseImpl::RS_SEQ_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_ADD, seqAdd(d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_SEQ_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_TRIAD,
                    seqTriad(d_a, d_b, d_c, chunkSize, scalar));
    break;

  /* GATHER KERNELS */
  case RSBaseImpl::RS_GATHER_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_COPY,
                    gatherCopy(d_a, d_b, d_c, d_idx1, chunkSize));
    break;

  case RSBaseImpl::RS_GATHER_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_SCALE,
                    gatherScale(d_a, d_b, d_c, d_idx1, chunkSize, scalar));
    break;

  case RSBaseImpl::RS_GATHER_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_ADD,
                    gatherAdd(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_GATHER_TRIAD:
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_GATHER_TRIAD,
        gatherTriad(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize, scalar));
    break;

  /* SCATTER KERNELS */
  case RSBaseImpl::RS_SCATTER_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_COPY,
                    scatterCopy(d_a, d_b, d_c, d_idx1, chunkSize));
    break;

  case RSBaseImpl::RS_SCATTER_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_SCALE,
                    scatterScale(d_a, d_b, d_c, d_idx1, chunkSize, scalar));
    break;

  case RSBaseImpl::RS_SCATTER_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_ADD,
                    scatterAdd(d_a, d_b, d_c, d_idx1, chunkSize));
    break;

  case RSBaseImpl::RS_SCATTER_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_TRIAD,
                    scatterTriad(d_a, d_b, d_c, d_idx1, chunkSize, scalar));
    break;

  /* SCATTER-GATHER KERNELS */
  case RSBaseImpl::RS_SG_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_COPY,
                    sgCopy(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_SG_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_SCALE,
                    sgScale(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize, scalar));
    break;

  case RSBaseImpl::RS_SG_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_ADD,
                    sgAdd(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize));
    break;

  case RSBaseImpl::RS_SG_TRIAD:
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SG_TRIAD,
        sgTriad(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize, scalar));
    break;

  /* CENTRAL KERNELS */
  case RSBaseImpl::RS_CENTRAL_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_COPY,
                    centralCopy(d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_CENTRAL_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_SCALE,
                    centralScale(d_a, d_b, d_c, chunkSize, scalar));
    break;

  case RSBaseImpl::RS_CENTRAL_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_ADD,
                    centralAdd(d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_CENTRAL_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_TRIAD,
                    centralTriad(d_a, d_b, d_c, chunkSize, scalar));
    break;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /* ALL KERNELS */
  case RSBaseImpl::RS_ALL:
    /* RS_SEQ_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_COPY, seqCopy(d_a, d_b, d_c, chunkSize));

    /* RS_SEQ_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_SCALE,
                    seqScale(d_a, d_b, d_c, chunkSize, scalar));

    /* RS_SEQ_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_ADD, seqAdd(d_a, d_b, d_c, chunkSize));

    /* RS_SEQ_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_TRIAD,
                    seqTriad(d_a, d_b, d_c, chunkSize, scalar));

    /* RS_GATHER_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_COPY,
                    gatherCopy(d_a, d_b, d_c, d_idx1, chunkSize));

    /* RS_GATHER_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_SCALE,
                    gatherScale(d_a, d_b, d_c, d_idx1, chunkSize, scalar));

    /* RS_GATHER_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_ADD,
                    gatherAdd(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));

    /* RS_GATHER_TRIAD */
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_GATHER_TRIAD,
        gatherTriad(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize, scalar));

    /* RS_SCATTER_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_COPY,
                    scatterCopy(d_a, d_b, d_c, d_idx1, chunkSize));

    /* RS_SCATTER_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_SCALE,
                    scatterScale(d_a, d_b, d_c, d_idx1, chunkSize, scalar));

    /* RS_SCATTER_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_ADD,
                    scatterAdd(d_a, d_b, d_c, d_idx1, chunkSize));

    /* RS_SCATTER_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_TRIAD,
                    scatterTriad(d_a, d_b, d_c, d_idx1, chunkSize, scalar));

    /* RS_SG_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_COPY,
                    sgCopy(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));

    /* RS_SG_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_SCALE,
                    sgScale(d_a, d_b, d_c, d_idx1, d_idx2, chunkSize, scalar));

    /* RS_SG_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_ADD,
                    sgAdd(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize));

    /* RS_SG_TRIAD */
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SG_TRIAD,
        sgTriad(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize, scalar));

    /* RS_CENTRAL_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_COPY,
                    centralCopy(d_a, d_b, d_c, chunkSize));

    /* RS_CENTRAL_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_SCALE,
                    centralScale(d_a, d_b, d_c, chunkSize, scalar));

    /* RS_CENTRAL_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_ADD,
                    centralAdd(d_a, d_b, d_c, chunkSize));

    /* RS_CENTRAL_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_TRIAD,
                    centralTriad(d_a, d_b, d_c, chunkSize, scalar));

    break;

  /* NO KERNELS, SOMETHING IS WRONG */
  default:
    if (myRank == 0) {
      std::cout << "RS_SHMEM_OACC::execute() - ERROR: KERNEL NOT SET"
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

#endif /* _RS_SHMEM_OACC_H_ */

/* EOF */
