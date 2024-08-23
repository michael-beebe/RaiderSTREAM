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
  int numGangs;
  int numWorkers;
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
  ssize_t scalar;

public:
  RS_OACC(const RSOpts &opts);

  ~RS_OACC();

  virtual bool setDevice() override;

  virtual bool allocateData() override;

  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  virtual bool freeData() override;
};

extern "C" {
double seqCopy(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
               ssize_t streamArraySize);

double seqScale(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
                ssize_t streamArraySize, double scalar);

double seqAdd(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
              ssize_t streamArraySize);

double seqTriad(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
                ssize_t streamArraySize, double scalar);

double gatherCopy(int ngangs, int nworkers, double *d_a, double *d_b,
                  double *d_c, ssize_t *d_IDX1, ssize_t streamArraySize);

double gatherScale(int ngangs, int nworkers, double *d_a, double *d_b,
                   double *d_c, ssize_t *d_IDX1, ssize_t streamArraySize,
                   double scalar);

double gatherAdd(int ngangs, int nworkers, double *d_a, double *d_b,
                 double *d_c, ssize_t *d_IDX1, ssize_t *d_IDX2,
                 ssize_t streamArraySize);

double gatherTriad(int ngangs, int nworkers, double *d_a, double *d_b,
                   double *d_c, ssize_t *d_IDX1, ssize_t *d_IDX2,
                   ssize_t streamArraySize, double scalar);

double scatterCopy(int ngangs, int nworkers, double *d_a, double *d_b,
                   double *d_c, ssize_t *d_IDX1, ssize_t streamArraySize);

double scatterScale(int ngangs, int nworkers, double *d_a, double *d_b,
                    double *d_c, ssize_t *d_IDX1, ssize_t streamArraySize,
                    double scalar);

double scatterAdd(int ngangs, int nworkers, double *d_a, double *d_b,
                  double *d_c, ssize_t *d_IDX1, ssize_t streamArraySize);

double scatterTriad(int ngangs, int nworkers, double *d_a, double *d_b,
                    double *d_c, ssize_t *d_IDX1, ssize_t streamArraySize,
                    double scalar);

double sgCopy(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
              ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t streamArraySize);

double sgScale(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
               ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t streamArraySize,
               double scalar);

double sgAdd(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
             ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t *d_IDX3,
             ssize_t streamArraySize);

double sgTriad(int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
               ssize_t *d_IDX1, ssize_t *d_IDX2, ssize_t *d_IDX3,
               ssize_t streamArraySize, double scalar);

double centralCopy(int ngangs, int nworkers, double *d_a, double *d_b,
                   double *d_c, ssize_t streamArraySize);

double centralScale(int ngangs, int nworkers, double *d_a, double *d_b,
                    double *d_c, ssize_t streamArraySize, double scalar);

double centralAdd(int ngangs, int nworkers, double *d_a, double *d_b,
                  double *d_c, ssize_t streamArraySize);

double centralTriad(int ngangs, int nworkers, double *d_a, double *d_b,
                    double *d_c, ssize_t streamArraySize, double scalar);
}

#endif /* _RS_OACC_H_ */
#endif /* _ENABLE_OACC_ */
