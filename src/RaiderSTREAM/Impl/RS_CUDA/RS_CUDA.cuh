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

#include <cuda_runtime.h>
#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief RaiderSTREAM CUDA implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using CUDA.
 */
class RS_CUDA : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
  int numPEs;
  int lArgc;
  char **lArgv;
  double *a;
  double *b;
  double *c;
  double *d_a;
  double *d_b;
  double *d_c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  ssize_t *d_idx1;
  ssize_t *d_idx2;
  ssize_t *d_idx3;
  double scalar;
  int threadBlocks;
  int threadsPerBlock;

public:
  RS_CUDA(const RSOpts& opts);

  ~RS_CUDA();

  vool printCudaDeviceProps();

  virtual bool allocateData() override;

  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool freeData() override;
};

__global__ void seqCopy(
  double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize
);
__global__ void seqScale(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t streamArraySize
);
__global__ void seqAdd(
  double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize
);
__global__ void seqTriad(
  double *d_a, double *d_b, double *d_c,
  double scalar, ssize_t streamArraySize
);
__global__ void gatherCopy(
  double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void gatherScale(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void gatherAdd(
  double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void gatherTriad(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void scatterCopy(
  double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void scatterScale(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void scatterAdd(
  double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void scatterTriad(
  double *d_a, double *d_b, double *d_c,
  double scalar, ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize
);
__global__ void sgCopy(
  double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
  ssize_t streamArraySize
);
__global__ void sgScale(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
  ssize_t streamArraySize
);
__global__ void sgAdd(
  double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
  ssize_t streamArraySize
);
__global__ void sgTriad(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
  ssize_t streamArraySize
);
__global__ void centralCopy(
  double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize
);
__global__ void centralScale(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t streamArraySize
);
__global__ void centralAdd(
  double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize
);
__global__ void centralTriad(
  double *d_a, double *d_b, double *d_c,
  double scalar,
  ssize_t streamArraySize
);

#endif /* _RS_CUDA_CUH_ */
#endif /* _ENABLE_CUDA_ */

/* EOF */
