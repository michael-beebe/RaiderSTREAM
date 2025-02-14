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
  long streamArraySize;
  int numPEs;
  // int numTeams;
  // int threadsPerTeam;
  int lArgc;
  char **lArgv;
  STREAM_TYPE scalar;
  STREAM_TYPE *d_a;
  STREAM_TYPE *d_b;
  STREAM_TYPE *d_c;
  ssize_t *d_idx1;
  ssize_t *d_idx2;
  ssize_t *d_idx3;
  int device;

public:
  RS_OMP_TARGET(const RSOpts &opts);

  ~RS_OMP_TARGET();

  virtual bool allocateData() override;

  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  virtual bool freeData() override;
};

extern "C" {
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t streamArraySize);

void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar);

void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
            ssize_t streamArraySize);

void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar);

void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t *IDX1, ssize_t streamArraySize);

void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t *IDX1, ssize_t streamArraySize, STREAM_TYPE scalar);

void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
               ssize_t *IDX1, ssize_t *IDX2, ssize_t streamArraySize);

void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t *IDX1, ssize_t *IDX2, ssize_t streamArraySize,
                 STREAM_TYPE scalar);

void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t *IDX1, ssize_t streamArraySize);

void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t *IDX1, ssize_t streamArraySize, STREAM_TYPE scalar);

void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t *IDX1, ssize_t streamArraySize);

void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t *IDX1, ssize_t streamArraySize, STREAM_TYPE scalar);

void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
            ssize_t *IDX1, ssize_t *IDX2, ssize_t streamArraySize);

void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t *IDX1, ssize_t *IDX2, ssize_t streamArraySize,
             STREAM_TYPE scalar);

void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
           ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
           ssize_t streamArraySize);

void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
             ssize_t streamArraySize, STREAM_TYPE scalar);

void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t streamArraySize);

void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar);

void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t streamArraySize);

void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar);
}

#endif /* _RS_OMP_TARGET_H_ */
#endif /* _ENABLE_OMP_TARGET_ */
