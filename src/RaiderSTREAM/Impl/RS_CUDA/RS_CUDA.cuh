/**
 * @file RS_CUDA.cuh
 * @brief CUDA header file for RaiderSTREAM benchmark implementation
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights Reserved.
 * @author michael.beebe@ttu.edu
 *
 * See LICENSE in the top level directory for licensing details
 */

#ifdef _ENABLE_CUDA_
#ifndef _RS_CUDA_CUH_
#define _RS_CUDA_CUH_

#include "RaiderSTREAM/RaiderSTREAM.h"
#include <cuda_runtime.h>

/**
 * @brief RaiderSTREAM CUDA implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using
 * CUDA.
 */
class RS_CUDA : public RSBaseImpl {
private:
  std::string kernelName;        /**< Name of the kernel being executed */
  ssize_t streamArraySize;       /**< Size of the stream arrays */
  ssize_t streamArrayMemSize;    /**< Memory size of stream arrays in bytes */
  int numPEs;                    /**< Number of processing elements */
  int lArgc;                     /**< Local copy of argc */
  char **lArgv;                  /**< Local copy of argv */
  STREAM_TYPE *a;                /**< Host array A */
  STREAM_TYPE *b;                /**< Host array B */
  STREAM_TYPE *c;                /**< Host array C */
  STREAM_TYPE *d_a;              /**< Device array A */
  STREAM_TYPE *d_b;              /**< Device array B */
  STREAM_TYPE *d_c;              /**< Device array C */
  ssize_t *idx1;                 /**< Host index array 1 */
  ssize_t *idx2;                 /**< Host index array 2 */
  ssize_t *idx3;                 /**< Host index array 3 */
  ssize_t *d_idx1;               /**< Device index array 1 */
  ssize_t *d_idx2;               /**< Device index array 2 */
  ssize_t *d_idx3;               /**< Device index array 3 */
  double scalar;                 /**< Scalar value used in operations */
  ssize_t idxArrayMemSize;       /**< Memory size of index arrays in bytes */
  int threadBlocks;              /**< Number of thread blocks */
  int threadsPerBlock;           /**< Number of threads per block */
  int deviceId;                  /**< CUDA device ID */

public:
  /**
   * @brief Constructor for RS_CUDA class
   * @param opts Configuration options for the implementation
   */
  RS_CUDA(const RSOpts &opts);

  /**
   * @brief Destructor for RS_CUDA class
   */
  ~RS_CUDA();

  /**
   * @brief Print CUDA device properties
   * @return True if successful, false otherwise
   */
  virtual bool printCudaDeviceProps();

  /**
   * @brief Allocate memory for arrays on host and device
   * @return True if successful, false otherwise
   */
  virtual bool allocateData(double * allocTime, double * initTime, double * randomGenTime) override;
 
  /**
   * @brief collect all results into one array
   *
   * @param collectTime The time taken to collect all results
  **/
  virtual void collectChunks(double *collectTime) override;

  /**
   * @brief Execute the benchmark kernels
   * @param[out] TIMES Array to store kernel execution times
   * @param[out] MBPS Array to store memory bandwidth results
   * @param[out] FLOPS Array to store floating point operations per second
   * @param[in] BYTES Array containing byte sizes for each kernel
   * @param[in] FLOATOPS Array containing floating point operation counts
   * @return True if successful, false otherwise
   */
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  /**
   * @brief Free allocated memory on host and device
   * @return True if successful, false otherwise
   */
  virtual bool freeData() override;
};

/**
 * @brief Sequential copy kernel
 * @param d_a Source array
 * @param d_b Unused array
 * @param d_c Destination array
 * @param streamArraySize Size of arrays
 */
__global__ void seqCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                        ssize_t streamArraySize);

/**
 * @brief Sequential scale kernel
 * @param d_a Unused array
 * @param d_b Destination array
 * @param d_c Source array
 * @param scalar Scaling factor
 * @param streamArraySize Size of arrays
 */
__global__ void seqScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                         ssize_t streamArraySize);

/**
 * @brief Sequential add kernel
 * @param d_a First source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param streamArraySize Size of arrays
 */
__global__ void seqAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                       ssize_t streamArraySize);

/**
 * @brief Sequential triad kernel
 * @param d_a Source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param scalar Scaling factor
 * @param streamArraySize Size of arrays
 */
__global__ void seqTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                         ssize_t streamArraySize);

/**
 * @brief Gather copy kernel
 * @param d_a Source array
 * @param d_b Unused array
 * @param d_c Destination array
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void gatherCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t *d_idx1, ssize_t *d_idx2,
                           ssize_t streamArraySize);

/**
 * @brief Gather scale kernel
 * @param d_a Unused array
 * @param d_b Destination array
 * @param d_c Source array
 * @param scalar Scaling factor
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void gatherScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);

/**
 * @brief Gather add kernel
 * @param d_a First source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void gatherAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                          ssize_t *d_idx1, ssize_t *d_idx2,
                          ssize_t streamArraySize);

/**
 * @brief Gather triad kernel
 * @param d_a Source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param scalar Scaling factor
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void gatherTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);

/**
 * @brief Scatter copy kernel
 * @param d_a Source array
 * @param d_b Unused array
 * @param d_c Destination array
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void scatterCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);

/**
 * @brief Scatter scale kernel
 * @param d_a Unused array
 * @param d_b Destination array
 * @param d_c Source array
 * @param scalar Scaling factor
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void scatterScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                             ssize_t streamArraySize);

/**
 * @brief Scatter add kernel
 * @param d_a First source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void scatterAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t *d_idx1, ssize_t *d_idx2,
                           ssize_t streamArraySize);

/**
 * @brief Scatter triad kernel
 * @param d_a Source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param scalar Scaling factor
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param streamArraySize Size of arrays
 */
__global__ void scatterTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                             ssize_t streamArraySize);

/**
 * @brief Scatter-gather copy kernel
 * @param d_a Source array
 * @param d_b Unused array
 * @param d_c Destination array
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param d_idx3 Third index array
 * @param streamArraySize Size of arrays
 */
__global__ void sgCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, ssize_t *d_idx1,
                       ssize_t *d_idx2, ssize_t *d_idx3,
                       ssize_t streamArraySize);

/**
 * @brief Scatter-gather scale kernel
 * @param d_a Unused array
 * @param d_b Destination array
 * @param d_c Source array
 * @param scalar Scaling factor
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param d_idx3 Third index array
 * @param streamArraySize Size of arrays
 */
__global__ void sgScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                        ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
                        ssize_t streamArraySize);

/**
 * @brief Scatter-gather add kernel
 * @param d_a First source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param d_idx3 Third index array
 * @param streamArraySize Size of arrays
 */
__global__ void sgAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, ssize_t *d_idx1,
                      ssize_t *d_idx2, ssize_t *d_idx3,
                      ssize_t streamArraySize);

/**
 * @brief Scatter-gather triad kernel
 * @param d_a Source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param scalar Scaling factor
 * @param d_idx1 First index array
 * @param d_idx2 Second index array
 * @param d_idx3 Third index array
 * @param streamArraySize Size of arrays
 */
__global__ void sgTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                        ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
                        ssize_t streamArraySize);

/**
 * @brief Central copy kernel
 * @param d_a Source array
 * @param d_b Unused array
 * @param d_c Destination array
 * @param streamArraySize Size of arrays
 */
__global__ void centralCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            ssize_t streamArraySize);

/**
 * @brief Central scale kernel
 * @param d_a Unused array
 * @param d_b Destination array
 * @param d_c Source array
 * @param scalar Scaling factor
 * @param streamArraySize Size of arrays
 */
__global__ void centralScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t streamArraySize);

/**
 * @brief Central add kernel
 * @param d_a First source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param streamArraySize Size of arrays
 */
__global__ void centralAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t streamArraySize);

/**
 * @brief Central triad kernel
 * @param d_a Source array
 * @param d_b Second source array
 * @param d_c Destination array
 * @param scalar Scaling factor
 * @param streamArraySize Size of arrays
 */
__global__ void centralTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t streamArraySize);

#endif /* _RS_CUDA_CUH_ */
#endif /* _ENABLE_CUDA_ */

/* EOF */
