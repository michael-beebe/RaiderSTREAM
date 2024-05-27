//
// _RS_SHMEM_OMP_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_SHMEM_OMP_
#ifndef _RS_SHMEM_OMP_H_
#define _RS_SHMEM_OMP_H_

#include <shmem.h>
#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

class RS_SHMEM_OMP : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
  int lArgc;
  char **lArgv;
  int numPEs;
  double *a;
  double *b;
  double *c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  ssize_t scalar;

public:
  RS_SHMEM_OMP(const RSOpts& opts);

  ~RS_SHMEM_OMP();

  virtual bool allocateData() override;
  
  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool freeData() override;
};

extern "C" { // FIXME: these might need to take in a `int numPEs` argument
  void seqCopy(
    double *a, double *b, double *c,
    ssize_t chunkSize
  );

  void seqScale(
    double *a, double *b, double *c,
    ssize_t chunkSize, double scalar
  );

  void seqAdd(
    double *a, double *b, double *c,
    ssize_t chunkSize
  );

  void seqTriad(
    double *a, double *b, double *c,
    ssize_t chunkSize, double scalar
  );

  void gatherCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t chunkSize
  );

  void gatherScale(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t chunkSize, double scalar
  );

  void gatherAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize
  );

  void gatherTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize, double scalar
  );

  void scatterCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t chunkSize
  );

  void scatterScale(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t chunkSize, double scalar
  );

  void scatterAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t chunkSize
  );

  void scatterTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t chunkSize, double scalar
  );

  void sgCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize
  );

  void sgScale(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize, double scalar
  );

  void sgAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t chunkSize
  );

  void sgTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t chunkSize, double scalar
  );

  void centralCopy(
    double *a, double *b, double *c,
    ssize_t chunkSize
  );

  void centralScale(
    double *a, double *b, double *c,
    ssize_t chunkSize,
    double scalar
  );

  void centralAdd(
    double *a, double *b, double *c,
    ssize_t chunkSize
  );

  void centralTriad(
    double *a, double *b, double *c,
    ssize_t chunkSize, double scalar
  );
}

#endif /* _RS_SHMEM_OMP_H_ */
#endif /* _ENABLE_SHMEM_OMP_ */
