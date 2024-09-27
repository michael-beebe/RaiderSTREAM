//
// _RS_CUDA_CUH_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

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
  std::string kernelName;
  ssize_t streamArraySize;
  ssize_t streamArrayMemSize;
  int numPEs;
  int lArgc;
  char **lArgv;
  STREAM_TYPE *a;
  STREAM_TYPE *b;
  STREAM_TYPE *c;
  STREAM_TYPE *d_a;
  STREAM_TYPE *d_b;
  STREAM_TYPE *d_c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  ssize_t *d_idx1;
  ssize_t *d_idx2;
  ssize_t *d_idx3;
  double scalar;
  ssize_t idxArrayMemSize;
  int threadBlocks;
  int threadsPerBlock;

public:
  RS_CUDA(const RSOpts &opts);

  ~RS_CUDA();

  virtual bool printCudaDeviceProps();

  virtual bool allocateData() override;

  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  virtual bool freeData() override;
};

__global__ void seqCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                        ssize_t streamArraySize);
__global__ void seqScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                         ssize_t streamArraySize);
__global__ void seqAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                       ssize_t streamArraySize);
__global__ void seqTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                         ssize_t streamArraySize);
__global__ void gatherCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t *d_idx1, ssize_t *d_idx2,
                           ssize_t streamArraySize);
__global__ void gatherScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);
__global__ void gatherAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                          ssize_t *d_idx1, ssize_t *d_idx2,
                          ssize_t streamArraySize);
__global__ void gatherTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);
__global__ void scatterCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            ssize_t *d_idx1, ssize_t *d_idx2,
                            ssize_t streamArraySize);
__global__ void scatterScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                             ssize_t streamArraySize);
__global__ void scatterAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t *d_idx1, ssize_t *d_idx2,
                           ssize_t streamArraySize);
__global__ void scatterTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t *d_idx1, ssize_t *d_idx2,
                             ssize_t streamArraySize);
__global__ void sgCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, ssize_t *d_idx1,
                       ssize_t *d_idx2, ssize_t *d_idx3,
                       ssize_t streamArraySize);
__global__ void sgScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                        ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
                        ssize_t streamArraySize);
__global__ void sgAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, ssize_t *d_idx1,
                      ssize_t *d_idx2, ssize_t *d_idx3,
                      ssize_t streamArraySize);
__global__ void sgTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c, STREAM_TYPE scalar,
                        ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
                        ssize_t streamArraySize);
__global__ void centralCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                            ssize_t streamArraySize);
__global__ void centralScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t streamArraySize);
__global__ void centralAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                           ssize_t streamArraySize);
__global__ void centralTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                             STREAM_TYPE scalar, ssize_t streamArraySize);

#endif /* _RS_CUDA_CUH_ */
#endif /* _ENABLE_CUDA_ */

/* EOF */
