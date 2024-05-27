//
// _RS_OMP_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_OMP_
#ifndef _RS_OMP_H_
#define _RS_OMP_H_

#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief RaiderSTREAM OpenMP implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using OpenMP.
 */
class RS_OMP : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
  int numPEs;
  int lArgc;
  char **lArgv;
  double *a;
  double *b;
  double *c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  ssize_t scalar;

public:
  RS_OMP(const RSOpts& opts);

  ~RS_OMP();

  virtual bool allocateData() override;

  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool freeData() override;
};

extern "C" {
  void seqCopy(
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void seqScale(
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );

  void seqAdd(
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void seqTriad(
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );

  void gatherCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  void gatherScale(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  void gatherAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize
  );

  void gatherTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize, double scalar
  );

  void scatterCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  void scatterScale(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  void scatterAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  void scatterTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  void sgCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize
  );

  void sgScale(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize, double scalar
  );

  void sgAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t streamArraySize
  );

  void sgTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t streamArraySize, double scalar
  );

  void centralCopy(
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void centralScale(
    double *a, double *b, double *c,
    ssize_t streamArraySize,
    double scalar
  );

  void centralAdd(
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void centralTriad(
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );
}

#endif /* _RS_OMP_H_ */
#endif /* _ENABLE_OMP_ */
