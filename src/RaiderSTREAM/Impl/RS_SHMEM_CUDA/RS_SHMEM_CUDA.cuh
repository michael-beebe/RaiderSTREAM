/**
 * @file RS_SHMEM_CUDA.cuh
 * @brief Header file for the RS_SHMEM_CUDA class implementing CUDA+OpenSHMEM STREAM benchmarks
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#ifdef _ENABLE_SHMEM_CUDA_
#ifndef _RS_SHMEM_CUDA_CUH_
#define _RS_SHMEM_CUDA_CUH_

#include "RaiderSTREAM/RaiderSTREAM.h"
#include <cuda_runtime.h>
#include <shmem.h>

/**
 * @brief RaiderSTREAM CUDA+OpenSHMEM implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using
 * CUDA with OpenSHMEM for distributed memory parallelism.
 */
class RS_SHMEM_CUDA : public RSBaseImpl {
private:
  std::string kernelName;         /**< Name of kernel being executed */
  ssize_t streamArraySize;        /**< Size of stream arrays */
  ssize_t streamArrayMemSize;     /**< Memory size of stream arrays in bytes */
  int numPEs;                     /**< Number of processing elements */
  int lArgc;                      /**< Local argument count */
  char **lArgv;                   /**< Local argument vector */
  STREAM_TYPE *a;                 /**< Host pointer for first stream array */
  STREAM_TYPE *b;                 /**< Host pointer for second stream array */ 
  STREAM_TYPE *c;                 /**< Host pointer for third stream array */
  STREAM_TYPE *d_a;               /**< Device pointer for first stream array */
  STREAM_TYPE *d_b;               /**< Device pointer for second stream array */
  STREAM_TYPE *d_c;               /**< Device pointer for third stream array */
  ssize_t *idx1;                  /**< Host pointer for first index array */
  ssize_t *idx2;                  /**< Host pointer for second index array */
  ssize_t *idx3;                  /**< Host pointer for third index array */
  ssize_t *d_idx1;                /**< Device pointer for first index array */
  ssize_t *d_idx2;                /**< Device pointer for second index array */
  ssize_t *d_idx3;                /**< Device pointer for third index array */
  double scalar;                  /**< Scalar value used in operations */
  ssize_t idxArrayMemSize;        /**< Memory size of index arrays in bytes */
  int threadBlocks;               /**< Number of thread blocks */
  int threadsPerBlock;            /**< Number of threads per block */
  int deviceId;                   /**< CUDA device ID */

public:
  /**
   * @brief Constructs an RS_SHMEM_CUDA object
   * @param opts Configuration options for the benchmark
   */
  RS_SHMEM_CUDA(const RSOpts &opts);

  /**
   * @brief Destroys the RS_SHMEM_CUDA object
   */
  ~RS_SHMEM_CUDA();

  /**
   * @brief Prints CUDA device properties
   * @return True if successful, false otherwise
   */
  virtual bool printCudaDeviceProps();

  /**
   * @brief Allocates and initializes memory for stream arrays
   * @return True if allocation succeeds, false otherwise
   */
  virtual bool allocateData() override;

  /**
   * @brief Executes the selected benchmark kernel
   * @param[out] TIMES Array to store execution times
   * @param[out] MBPS Array to store bandwidth measurements
   * @param[out] FLOPS Array to store floating point operations per second
   * @param[out] BYTES Array to store bytes transferred
   * @param[out] FLOATOPS Array to store floating point operations performed
   * @return True if execution succeeds, false otherwise
   */
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  /**
   * @brief Frees allocated memory
   * @return True if deallocation succeeds, false otherwise
   */
  virtual bool freeData() override;
};

/**
 * @brief Sequential copy kernel
 * @param[in] d_a Source array
 * @param[in] d_b Unused array
 * @param[out] d_c Destination array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void seqCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                        ssize_t streamArraySize);

/**
 * @brief Sequential scale kernel
 * @param[in] d_a Unused array
 * @param[out] d_b Destination array
 * @param[in] d_c Source array
 * @param[in] scalar Scaling factor
 * @param[in] streamArraySize Size of arrays
 */
__global__ void seqScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                         STREAM_TYPE scalar, ssize_t streamArraySize);

/**
 * @brief Sequential add kernel
 * @param[in] d_a First source array
 * @param[in] d_b Second source array
 * @param[out] d_c Destination array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void seqAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                       ssize_t streamArraySize);

/**
 * @brief Sequential triad kernel
 * @param[out] d_a Destination array
 * @param[in] d_b First source array
 * @param[in] d_c Second source array
 * @param[in] scalar Scaling factor
 * @param[in] streamArraySize Size of arrays
 */
__global__ void seqTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                         STREAM_TYPE scalar, ssize_t streamArraySize);

/**
 * @brief Gather copy kernel
 * @param[in] d_a Source array
 * @param[in] d_b Unused array
 * @param[out] d_c Destination array
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void gatherCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t *d_idx1, ssize_t *d_idx2,
                           ssize_t streamArraySize);

/**
 * @brief Gather scale kernel
 * @param[in] d_a Unused array
 * @param[out] d_b Destination array
 * @param[in] d_c Source array
 * @param[in] scalar Scaling factor
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void gatherScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                            STREAM_TYPE *d_c, STREAM_TYPE scalar,
                            ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);

/**
 * @brief Gather add kernel
 * @param[in] d_a First source array
 * @param[in] d_b Second source array
 * @param[out] d_c Destination array
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void gatherAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                          ssize_t *d_idx1, ssize_t *d_idx2,
                          ssize_t streamArraySize);

/**
 * @brief Gather triad kernel
 * @param[out] d_a Destination array
 * @param[in] d_b First source array
 * @param[in] d_c Second source array
 * @param[in] scalar Scaling factor
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void gatherTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                            STREAM_TYPE *d_c, STREAM_TYPE scalar,
                            ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);

/**
 * @brief Scatter copy kernel
 * @param[in] d_a Source array
 * @param[in] d_b Unused array
 * @param[out] d_c Destination array
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void scatterCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                            STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);

/**
 * @brief Scatter scale kernel
 * @param[in] d_a Unused array
 * @param[out] d_b Destination array
 * @param[in] d_c Source array
 * @param[in] scalar Scaling factor
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void scatterScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                             STREAM_TYPE *d_c, STREAM_TYPE scalar,
                             ssize_t *d_idx1, ssize_t *d_idx2,
                             ssize_t streamArraySize);

/**
 * @brief Scatter add kernel
 * @param[in] d_a First source array
 * @param[in] d_b Second source array
 * @param[out] d_c Destination array
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void scatterAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t *d_idx1, ssize_t *d_idx2,
                           ssize_t streamArraySize);

/**
 * @brief Scatter triad kernel
 * @param[out] d_a Destination array
 * @param[in] d_b First source array
 * @param[in] d_c Second source array
 * @param[in] scalar Scaling factor
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void scatterTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                             STREAM_TYPE *d_c, STREAM_TYPE scalar,
                             ssize_t *d_idx1, ssize_t *d_idx2,
                             ssize_t streamArraySize);

/**
 * @brief Scatter-gather copy kernel
 * @param[in] d_a Source array
 * @param[in] d_b Unused array
 * @param[out] d_c Destination array
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] d_idx3 Third index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void sgCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                       ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
                       ssize_t streamArraySize);

/**
 * @brief Scatter-gather scale kernel
 * @param[in] d_a Unused array
 * @param[out] d_b Destination array
 * @param[in] d_c Source array
 * @param[in] scalar Scaling factor
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] d_idx3 Third index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void sgScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                        STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                        ssize_t *d_idx3, ssize_t streamArraySize);

/**
 * @brief Scatter-gather add kernel
 * @param[in] d_a First source array
 * @param[in] d_b Second source array
 * @param[out] d_c Destination array
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] d_idx3 Third index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void sgAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                      ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
                      ssize_t streamArraySize);

/**
 * @brief Scatter-gather triad kernel
 * @param[out] d_a Destination array
 * @param[in] d_b First source array
 * @param[in] d_c Second source array
 * @param[in] scalar Scaling factor
 * @param[in] d_idx1 First index array
 * @param[in] d_idx2 Second index array
 * @param[in] d_idx3 Third index array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void sgTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                        STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                        ssize_t *d_idx3, ssize_t streamArraySize);

/**
 * @brief Central copy kernel using a single location
 * @param[in] d_a Source array
 * @param[in] d_b Unused array
 * @param[out] d_c Destination array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void centralCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                            STREAM_TYPE *d_c, ssize_t streamArraySize);

/**
 * @brief Central scale kernel using a single location
 * @param[in] d_a Unused array
 * @param[out] d_b Destination array
 * @param[in] d_c Source array
 * @param[in] scalar Scaling factor
 * @param[in] streamArraySize Size of arrays
 */
__global__ void centralScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                             STREAM_TYPE *d_c, STREAM_TYPE scalar,
                             ssize_t streamArraySize);

/**
 * @brief Central add kernel using a single location
 * @param[in] d_a First source array
 * @param[in] d_b Second source array
 * @param[out] d_c Destination array
 * @param[in] streamArraySize Size of arrays
 */
__global__ void centralAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t streamArraySize);

/**
 * @brief Central triad kernel using a single location
 * @param[out] d_a Destination array
 * @param[in] d_b First source array
 * @param[in] d_c Second source array
 * @param[in] scalar Scaling factor
 * @param[in] streamArraySize Size of arrays
 */
__global__ void centralTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                             STREAM_TYPE *d_c, STREAM_TYPE scalar,
                             ssize_t streamArraySize);

#endif /* _RS_CUDA_CUH_ */
#endif /* _ENABLE_CUDA_ */

/* EOF */
