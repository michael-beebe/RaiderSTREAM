//
// _RS_MPI_OMP_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_MPI_OMP_
#ifndef _RS_MPI_OMP_H_
#define _RS_MPI_OMP_H_

#include <mpi.h>
#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

class RS_MPI_OMP : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
  int lArgc;
  char **lArgv;
  int numPEs;
  STREAM_TYPE *a;
  STREAM_TYPE *b;
  STREAM_TYPE *c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  ssize_t scalar;

public:
  RS_MPI_OMP(const RSOpts& opts);

  ~RS_MPI_OMP();

  virtual bool allocateData() override;

  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool freeData() override;
};

extern "C" { // FIXME: these might need to take in a `int numPEs` argument
  void seqCopy(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize
  );

  void seqScale(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize, double scalar
  );

  void seqAdd(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize
  );

  void seqTriad(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize, double scalar
  );

  void gatherCopy(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1,
    ssize_t chunkSize
  );

  void gatherScale(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1,
    ssize_t chunkSize, double scalar
  );

  void gatherAdd(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize
  );

  void gatherTriad(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize, double scalar
  );

  void scatterCopy(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1,
    ssize_t chunkSize
  );

  void scatterScale(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1,
    ssize_t chunkSize, double scalar
  );

  void scatterAdd(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1,
    ssize_t chunkSize
  );

  void scatterTriad(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1,
    ssize_t chunkSize, double scalar
  );

  void sgCopy(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize
  );

  void sgScale(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t chunkSize, double scalar
  );

  void sgAdd(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t chunkSize
  );

  void sgTriad(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t chunkSize, double scalar
  );

  void centralCopy(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize
  );

  void centralScale(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize,
    double scalar
  );

  void centralAdd(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize
  );

  void centralTriad(
    STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
    ssize_t chunkSize, double scalar
  );
}

#endif /* _RS_MPI_OMP_H_ */
#endif /* _ENABLE_MPI_OMP_ */
