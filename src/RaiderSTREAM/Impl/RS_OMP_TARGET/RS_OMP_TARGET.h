//
// _RS_OMP_TARGET_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_OMP_TARGET_
#ifndef _RS_OMP_TARGET_H_
#define _RS_OMP_TARGET_H_

#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief RaiderSTREAM OpenMP Target implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark
 * using the OpenMP target pragma option.
 */
class RS_OMP_TARGET : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
  int numPEs;
  int numTeams;
  int threadsPerTeam;
  int lArgc;
  char **lArgv;
  ssize_t scalar;
  double *a;
  double *b;
  double *c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;

public:
  RS_OMP_TARGET(const RSOpts& opts);

  ~RS_OMP_TARGET();

  void prepare();

  virtual bool allocateData() override;

  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool freeData() override;
};

extern "C" {
  double seqCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  double seqScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );

  double seqAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  double seqTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );

  double gatherCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  double gatherScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  double gatherAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize
  );

  double gatherTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize, double scalar
  );

  double scatterCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  double scatterScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  double scatterAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  double scatterTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  double sgCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize
  );

  double sgScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize, double scalar
  );

  double sgAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t streamArraySize
  );

  double sgTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t streamArraySize, double scalar
  );

  double centralCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  double centralScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize,
    double scalar
  );

  double centralAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  double centralTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );
}

#endif /* _RS_OMP_TARGET_H_ */
#endif /* _ENABLE_OMP_TARGET_ */
