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
  double *a;
  double *b;
  double *c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  ssize_t scalar;

public:
  RS_OMP_TARGET(const RSOpts& opts);

  ~RS_OMP_TARGET();

  virtual bool allocateData() override;

  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool freeData() override;
};

extern "C" {
  void seqCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void seqScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );

  void seqAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void seqTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );

  void gatherCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  void gatherScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  void gatherAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize
  );

  void gatherTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize, double scalar
  );

  void scatterCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  void scatterScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  void scatterAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize
  );

  void scatterTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t streamArraySize, double scalar
  );

  void sgCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize
  );

  void sgScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t streamArraySize, double scalar
  );

  void sgAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t streamArraySize
  );

  void sgTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t streamArraySize, double scalar
  );

  void centralCopy(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void centralScale(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize,
    double scalar
  );

  void centralAdd(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize
  );

  void centralTriad(
    int nteams, int threads,
    double *a, double *b, double *c,
    ssize_t streamArraySize, double scalar
  );
}

#endif /* _RS_OMP_TARGET_H_ */
#endif /* _ENABLE_OMP_TARGET_ */
