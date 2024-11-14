//
// _RS_CUDA_CU_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_SHMEM_CUDA_

#include "RS_SHMEM_CUDA.cuh"

/* This sanitycheck is used to prevent
 * the compiler from getting "too slick
 * with it" when it comes to reasoning
 * about our code. This copy is done
 * outside of benchmark time recording.
 */
#define CUDA_SANITYCHECK                                                       \
  do {                                                                         \
    cudaMemcpy(a, d_a, streamArraySize, cudaMemcpyDeviceToHost);               \
    cudaMemcpy(b, d_b, streamArraySize, cudaMemcpyDeviceToHost);               \
    cudaMemcpy(c, d_c, streamArraySize, cudaMemcpyDeviceToHost);               \
    acc = 0;                                                                   \
    for (ssize_t i = 1; i < streamArraySize; i *= 2) {                         \
      acc += a[i] + b[i] + c[i];                                               \
    }                                                                          \
    std::cout << "Compiler sanity check: " << acc << std::endl;                \
  } while (false)

#ifdef _SHMEM_1_5_
#define SHMEM_BENCHMARK(k, ...)                                                \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    shmem_barrier_all();                                                       \
    startTime = mySecond();                                                    \
    __VA_ARGS__;                                                               \
    cudaDeviceSynchronize();                                                   \
    shmem_barrier_all();                                                       \
    endTime = mySecond();                                                      \
    CUDA_SANITYCHECK;                                                          \
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
    cudaDeviceSynchronize();                                                   \
    shmem_barrier_all();                                                       \
    startTime = mySecond();                                                    \
    f;                                                                         \
    cudaDeviceSynchronize();                                                   \
    shmem_barrier_all();                                                       \
    endTime = mySecond();                                                      \
    runTime = calculateRunTime(startTime, endTime);                            \
    mbps = calculateMBPS(BYTES[k], runTime);                                   \
    flops = calculateFLOPS(FLOATOPS[k], runTime);                              \
    shmem_double_sum_to_all(totalRunTime, &runTime, 1, 0, 0, size, pWrk,       \
                            pSync);                                            \
    shmem_double_sum_to_all(totalMbps, &mbps, 1, 0, 0, size, pWrk, pSync);     \
    shmem_double_sum_to_all(totalFlops, &flops, 1, 0, 0, size, pWrk, pSync);   \
    if (myRank == 0) {                                                         \
      TIMES[k] = *totalRunTime / size;                                         \
      MBPS[k] = *totalMbps / size;                                             \
      FLOPS[k] = *totalFlops / size;                                           \
    }                                                                          \
  } while (false)
#endif

/**************************************************
 * @brief Constructor for the RS_CUDA class.
 *
 * Initializes the RS_CUDA object with the specified options.
 *
 * @param opts Options for the RS_CUDA object.
 **************************************************/
RS_SHMEM_CUDA::RS_SHMEM_CUDA(const RSOpts &opts)
    : RSBaseImpl("RS_SHMEM_CUDA",
                 opts.getKernelTypeFromName(opts.getKernelName())),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()), streamArrayMemSize(0),
      idxArrayMemSize(0), numPEs(opts.getNumPEs()), lArgc(0), lArgv(nullptr),
      a(nullptr), b(nullptr), c(nullptr), d_a(nullptr), d_b(nullptr),
      d_c(nullptr), idx1(nullptr), idx2(nullptr), idx3(nullptr),
      d_idx1(nullptr), d_idx2(nullptr), d_idx3(nullptr), scalar(3.0),
      threadBlocks(opts.getThreadBlocks()),
      threadsPerBlock(opts.getThreadsPerBlocks()), deviceId(opts.getDeviceId()) {}

RS_SHMEM_CUDA::~RS_SHMEM_CUDA() {}

/********************************************
 * @brief Print basic info about CUDA device.
 *
 * Currently unimplemented.
 *
 * @return If info was obtained successfuly.
 ********************************************/
bool RS_SHMEM_CUDA::printCudaDeviceProps() {
  // TODO:
  return true;
}

/**********************************************
 * @brief Allocates and initializes memory for
 *        data arrays and copies data to the device.
 *
 * @return True if allocation and copy are
 *         successful, false otherwise.
 **********************************************/
bool RS_SHMEM_CUDA::allocateData() {
  if(cudaSetDevice(deviceId) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::allocateData() - ERROR: failed setting CUDA device to "
              << deviceId
              << std::endl;
  }
  int myRank = shmem_my_pe(); /* Current rank */
  int size = shmem_n_pes();   /* Number of shmem ranks */

  if (numPEs == 0) {
    std::cout << "RS_SHMEM_CUDA::allocateData() - ERROR: 'pes' cannot be 0"
              << std::endl;
    return false;
  }

  shmem_barrier_all();

  /* If updated, also update corresponding
   * region in RS_SHMEM_CUDA::execute. */
  /* Calculate the chunk size for each rank */
  ssize_t chunkSize = streamArraySize / size;
  ssize_t remainder = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if (myRank == size - 1) {
    chunkSize += remainder;
  }

  if (threadBlocks <= 0) {
    std::cout
        << "RS_SHMEM_CUDA::AllocateData: threadBlocks must be greater than 0"
        << std::endl;
    return false;
  }
  if (threadsPerBlock <= 0) {
    std::cout
        << "RS_SHMEM_CUDA::AllocateData: threadsPerBlock must be greater than 0"
        << std::endl;
    return false;
  }

  /* Allocate host memory */
  a = new STREAM_TYPE[chunkSize];
  b = new STREAM_TYPE[chunkSize];
  c = new STREAM_TYPE[chunkSize];
  idx1 = new ssize_t[chunkSize];
  idx2 = new ssize_t[chunkSize];
  idx3 = new ssize_t[chunkSize];

  for (ssize_t i = 0; i < chunkSize; i++) {
    a[i] = b[i] = c[i] = i;
  }

  streamArrayMemSize = chunkSize * sizeof(STREAM_TYPE);
  idxArrayMemSize = chunkSize * sizeof(ssize_t);

#ifdef _ARRAYGEN_
  initReadIdxArray(idx1, chunkSize, "RaiderSTREAM/arraygen/IDX1.txt");
  initReadIdxArray(idx2, chunkSize, "RaiderSTREAM/arraygen/IDX2.txt");
  initReadIdxArray(idx3, chunkSize, "RaiderSTREAM/arraygen/IDX3.txt");
#else
  initRandomIdxArray(idx1, chunkSize);
  initRandomIdxArray(idx2, chunkSize);
  initRandomIdxArray(idx3, chunkSize);
#endif

  /* a -> d_a */
  if (cudaMalloc(&d_a, streamArrayMemSize) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_a' could not be allocated "
                 "on device"
              << std::endl;
    cudaFree(d_a);
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  if (cudaMemcpy(d_a, a, streamArrayMemSize, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout
        << "RS_SHMEM_CUDA::AllocateData : 'd_a' could not be copied to device"
        << std::endl;
    return false;
  }

  /* b -> d_b */
  if (cudaMalloc(&d_b, streamArrayMemSize) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_b' could not be allocated "
                 "on device"
              << std::endl;
    cudaFree(d_b);
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  if (cudaMemcpy(d_b, b, streamArrayMemSize, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout
        << "RS_SHMEM_CUDA::AllocateData : 'd_b' could not be copied to device"
        << std::endl;
    return false;
  }

  /* c -> d_c */
  if (cudaMalloc(&d_c, streamArrayMemSize) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_c' could not be allocated "
                 "on device"
              << std::endl;
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  if (cudaMemcpy(d_c, c, streamArrayMemSize, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout
        << "RS_SHMEM_CUDA::AllocateData : 'd_c' could not be copied to device"
        << std::endl;
    return false;
  }

  /* idx1 -> d_idx1 */
  if (cudaMalloc(&d_idx1, idxArrayMemSize) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_idx1' could not be "
                 "allocated on device"
              << std::endl;
    cudaFree(d_idx1);
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  if (cudaMemcpy(d_idx1, idx1, idxArrayMemSize, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_idx1' could not be copied "
                 "to device"
              << std::endl;
    return false;
  }

  /* idx2 -> d_idx2 */
  if (cudaMalloc(&d_idx2, idxArrayMemSize) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_idx2' could not be "
                 "allocated on device"
              << std::endl;
    cudaFree(d_idx2);
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  if (cudaMemcpy(d_idx2, idx2, idxArrayMemSize, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_idx2' could not be copied "
                 "to device"
              << std::endl;
    return false;
  }

  /* idx3 -> d_idx3 */
  if (cudaMalloc(&d_idx3, idxArrayMemSize) != cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_idx3' could not be "
                 "allocated on device"
              << std::endl;
    cudaFree(d_idx3);
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  if (cudaMemcpy(d_idx3, idx3, idxArrayMemSize, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout << "RS_SHMEM_CUDA::AllocateData : 'd_idx3' could not be copied "
                 "to device"
              << std::endl;
    return false;
  }

  STREAM_TYPE acc = 0.0;
  for (ssize_t i = 1; i < chunkSize; i *= 2) {
    acc += a[i] + b[i] + c[i];
  }
  std::cout << "Compiler sanity check: " << acc << std::endl;

#ifdef _DEBUG_
  std::cout << "==============================================================="
               "===================="
            << std::endl;
  std::cout << " RaiderSTREAM Array Info:" << std::endl;
  std::cout << "==============================================================="
               "===================="
            << std::endl;
  std::cout << "chunkSize         = " << chunkSize << std::endl;
  std::cout << "a[chunkSize-1]    = " << a[chunkSize - 1] << std::endl;
  std::cout << "b[chunkSize-1]    = " << b[chunkSize - 1] << std::endl;
  std::cout << "c[chunkSize-1]    = " << c[chunkSize - 1] << std::endl;
  std::cout << "idx1[chunkSize-1] = " << idx1[chunkSize - 1] << std::endl;
  std::cout << "idx2[chunkSize-1] = " << idx2[chunkSize - 1] << std::endl;
  std::cout << "idx3[chunkSize-1] = " << idx3[chunkSize - 1] << std::endl;
  std::cout << "==============================================================="
               "===================="
            << std::endl;
#endif

  return true;
}

/**************************************************
 * @brief Frees all allocated memory for the
 *        RS_SHMEM_CUDA object.
 *
 * This function deallocates memory for both host
 * and device pointers.
 *
 * @return true if all memory was successfully freed.
 **************************************************/
bool RS_SHMEM_CUDA::freeData() {
  if (a) {
    delete[] a;
  }
  if (b) {
    delete[] b;
  }
  if (c) {
    delete[] c;
  }
  if (idx1) {
    delete[] idx1;
  }
  if (idx2) {
    delete[] idx2;
  }
  if (idx3) {
    delete[] idx3;
  }
  if (d_a) {
    cudaFree(d_a);
  }
  if (d_b) {
    cudaFree(d_b);
  }
  if (d_c) {
    cudaFree(d_c);
  }
  if (d_idx1) {
    cudaFree(d_idx1);
  }
  if (d_idx2) {
    cudaFree(d_idx2);
  }
  if (d_idx3) {
    cudaFree(d_idx3);
  }
  return true;
}

/**************************************************
 * @brief Executes the specified kernel using CUDA.
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
bool RS_SHMEM_CUDA::execute(double *TIMES, double *MBPS, double *FLOPS,
                            double *BYTES, double *FLOATOPS) {
  double startTime = 0.0;
  double endTime = 0.0;
  double runTime = 0.0;
  double mbps = 0.0;
  double flops = 0.0;
  double *totalRunTime = (double *)shmem_malloc(sizeof(double));
  double *totalMbps = (double *)shmem_malloc(sizeof(double));
  double *totalFlops = (double *)shmem_malloc(sizeof(double));
  STREAM_TYPE acc;

  int myRank = shmem_my_pe(); /* Current rank */
  int size = shmem_n_pes();   /* Number of shmem ranks */
  size_t syncSize = SHMEM_SYNC_SIZE;

#ifdef _SHMEM_1_4_
  double *pWrk = static_cast<double *>(shmem_malloc(size * sizeof(double));
  long *pSync = static_cast<long *>(shmem_malloc(syncSize * sizeof(double));
  for(size_t i = 0; i < syncSize; ++i)
    pSync[i] = SHMEM_SYNC_SIZE;
#endif

  if (numPEs == 0) {
    std::cout << "RS_SHMEM_CUDA::execute() - ERROR: 'pes' cannot be 0"
              << std::endl;
    return false;
  }

  shmem_barrier_all();

  /* If updated, also update corresponding
   * region in RS_SHMEM_CUDA::allocateData. */
  /* Calculate the chunk size for each rank */
  ssize_t chunkSize = streamArraySize / size;
  ssize_t remainder = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if (myRank == size - 1) {
    chunkSize += remainder;
  }

  /* cuda likes to be too smart for its
   * own good, and will delay certain init
   * work until the device is needed.
   * run a kernel and throw away the results
   * to force initialization
   */
  cudaDeviceSynchronize();
  sgCopy<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c, d_idx1, d_idx2,
                                            d_idx3, chunkSize);
  cudaDeviceSynchronize();
  shmem_barrier_all();

  RSBaseImpl::RSKernelType kType = getKernelType();

  switch (kType) {
  /* Sequential Kernels */
  case RSBaseImpl::RS_SEQ_COPY:
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SEQ_COPY,
        seqCopy<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_SEQ_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_SCALE,
                    seqScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));
    break;

  case RSBaseImpl::RS_SEQ_ADD:
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SEQ_ADD,
        seqAdd<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_SEQ_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_TRIAD,
                    seqTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));
    break;

  /* Gather kernels */
  case RSBaseImpl::RS_GATHER_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_COPY,
                    gatherCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_GATHER_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_SCALE,
                    gatherScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_GATHER_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_ADD,
                    gatherAdd<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_GATHER_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_TRIAD,
                    gatherTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));
    break;

  /* Scatter kernels */
  case RSBaseImpl::RS_SCATTER_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_COPY,
                    scatterCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_SCATTER_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_SCALE,
                    scatterScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_SCATTER_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_ADD,
                    scatterAdd<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));
    break;

  case RSBaseImpl::RS_SCATTER_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_TRIAD,
                    scatterTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));
    break;

  /* Scatter-Gather kernels */
  case RSBaseImpl::RS_SG_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_COPY,
                    sgCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize));
    break;

  case RSBaseImpl::RS_SG_SCALE:
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SG_SCALE,
        sgScale<<<threadBlocks, threadsPerBlock>>>(
            d_a, d_b, d_c, scalar, d_idx1, d_idx2, d_idx3, chunkSize));
    break;

  case RSBaseImpl::RS_SG_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_ADD,
                    sgAdd<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize));
    break;

  case RSBaseImpl::RS_SG_TRIAD:
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SG_TRIAD,
        sgTriad<<<threadBlocks, threadsPerBlock>>>(
            d_a, d_b, d_c, scalar, d_idx1, d_idx2, d_idx3, chunkSize));
    break;

  /* Central kernels */
  case RSBaseImpl::RS_CENTRAL_COPY:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_COPY,
                    centralCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, chunkSize));
    break;

  case RSBaseImpl::RS_CENTRAL_SCALE:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_SCALE,
                    centralScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));
    break;

  case RSBaseImpl::RS_CENTRAL_ADD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_ADD,
                    centralAdd<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c,
                                                                  chunkSize));
    break;

  case RSBaseImpl::RS_CENTRAL_TRIAD:
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_TRIAD,
                    centralTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));
    break;

  /* All kernels */
  case RSBaseImpl::RS_ALL:
    /* RS_SEQ_COPY */
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SEQ_COPY,
        seqCopy<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c, chunkSize));

    /* RS_SEQ_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_SCALE,
                    seqScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));
    /* RS_SEQ_ADD */
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SEQ_ADD,
        seqAdd<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c, chunkSize));

    /* RS_SEQ_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SEQ_TRIAD,
                    seqTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));

    /* RS_GATHER_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_COPY,
                    gatherCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));

    /* RS_GATHER_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_SCALE,
                    gatherScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));

    /* RS_GATHER_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_ADD,
                    gatherAdd<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));

    /* RS_GATHER_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_GATHER_TRIAD,
                    gatherTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));

    /* RS_SCATTER_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_COPY,
                    scatterCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));

    /* RS_SCATTER_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_SCALE,
                    scatterScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));

    /* RS_SCATTER_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_ADD,
                    scatterAdd<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, chunkSize));

    /* RS_SCATTER_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SCATTER_TRIAD,
                    scatterTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, d_idx1, d_idx2, chunkSize));

    /* RS_SG_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_COPY,
                    sgCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize));

    /* RS_SG_SCALE */
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SG_SCALE,
        sgScale<<<threadBlocks, threadsPerBlock>>>(
            d_a, d_b, d_c, scalar, d_idx1, d_idx2, d_idx3, chunkSize));

    /* RS_SG_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_SG_ADD,
                    sgAdd<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, chunkSize));

    /* RS_SG_TRIAD */
    SHMEM_BENCHMARK(
        RSBaseImpl::RS_SG_TRIAD,
        sgTriad<<<threadBlocks, threadsPerBlock>>>(
            d_a, d_b, d_c, scalar, d_idx1, d_idx2, d_idx3, chunkSize));

    /* RS_CENTRAL_COPY */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_COPY,
                    centralCopy<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, chunkSize));

    /* RS_CENTRAL_SCALE */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_SCALE,
                    centralScale<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));

    /* RS_CENTRAL_ADD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_ADD,
                    centralAdd<<<threadBlocks, threadsPerBlock>>>(d_a, d_b, d_c,
                                                                  chunkSize));

    /* RS_CENTRAL_TRIAD */
    SHMEM_BENCHMARK(RSBaseImpl::RS_CENTRAL_TRIAD,
                    centralTriad<<<threadBlocks, threadsPerBlock>>>(
                        d_a, d_b, d_c, scalar, chunkSize));
    break;

  /* No kernel, something is wrong */
  default:
    std::cout << "RS_SHMEM_CUDA::execute() - ERROR: KERNEL NOT SET"
              << std::endl;

#ifdef _SHMEM_1_4_
	shmem_free(pWrk);
	shmem_free(pSync);
#endif
    return false;
  }

  CUDA_SANITYCHECK;


#ifdef _SHMEM_1_4_
	shmem_free(pWrk);
	shmem_free(pSync);
#endif

  return true;
}

#endif /* _ENABLE_SHMEM_CUDA_ */

/* EOF */
