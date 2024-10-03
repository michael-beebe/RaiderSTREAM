//
// _RS_OACC_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_OACC_
#ifndef _RS_OACC_H_
#define _RS_OACC_H_

#include <openacc.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief RaiderSTREAM OpenACC implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using
 * OpenMP.
 */
class RS_OACC : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
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
  STREAM_TYPE scalar;

public:
  RS_OACC(const RSOpts &opts);

  ~RS_OACC();

  virtual bool setDevice();

  virtual bool allocateData() override;

  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  virtual bool freeData() override;
};

extern "C" {
double seqCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
               ssize_t streamArraySize);

double seqScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                ssize_t streamArraySize, STREAM_TYPE scalar);

double seqAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
              ssize_t streamArraySize);

double seqTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                ssize_t streamArraySize, STREAM_TYPE scalar);

double gatherCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                  STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t streamArraySize);

double gatherScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t streamArraySize,
                   STREAM_TYPE scalar);

double gatherAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                 STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t *d_IDX2,
                 ssize_t streamArraySize);

double gatherTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t *d_IDX2,
                   ssize_t streamArraySize, STREAM_TYPE scalar);

double scatterCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t streamArraySize);

double scatterScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t streamArraySize,
                    STREAM_TYPE scalar);

double scatterAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                  STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t streamArraySize);

double scatterTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t *d_IDX1, ssize_t streamArraySize,
                    STREAM_TYPE scalar);

double sgCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
              ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t streamArraySize);

double sgScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
               ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t streamArraySize,
               STREAM_TYPE scalar);

double sgAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
             ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t *d_IDX3,
             ssize_t streamArraySize);

double sgTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
               ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t *d_IDX3,
               ssize_t streamArraySize, STREAM_TYPE scalar);

double centralCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t streamArraySize);

double centralScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t streamArraySize, STREAM_TYPE scalar);

double centralAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                  STREAM_TYPE *d_c, ssize_t streamArraySize);

double centralTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t streamArraySize, STREAM_TYPE scalar);
}

#endif /* _RS_OACC_H_ */
#endif /* _ENABLE_OACC_ */
