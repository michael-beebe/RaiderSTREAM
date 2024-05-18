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

#include "RaiderSTREAM/RSBaseImpl.h"
#include "RaiderSTREAM/RSOpts.h"

/**
 * @brief RaiderSTREAM OpenMP implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using OpenMP.
 */
class RS_OMP : public RSBaseImpl {
private:
  double *a;
  double *b;
  double *c;
  ssize_t *idx1;
  ssize_t *idx2;
  ssize_t *idx3;
  int scalar;

  // command line options
  std::string kernelName;
  ssize_t streamArraySize;
  int numTimes;
  std::string streamType;
  int numPEs;
  int lArgc;
  char **lArgv;

public:
  // RaiderSTREAM OMP constructor
  RS_OMP(const RSOpts& opts);
  // RS_OMP(
  //   std::string implName,
  //   RSBaseImpl::RSKernelType kType
  // );

  // RaiderSTREAM OMP destructor
  ~RS_OMP();

  // RaiderSTREAM OMP execute
  virtual bool execute( double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS ) override;
  // virtual bool execute(
  //   double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS,
  //   double *a, double *b, double *c, double *idx1, double *idx2, double *idx3, double scalar
  // ) override;

  virtual bool allocateData(
    double *a, double *b, double *c,
    ssize_t *idx1, ssize_t *idx2, ssize_t *idx3
  ) override;

  virtual bool freeData() override;
};

extern "C" {
  void seqCopy(
    double *a, double *b, double *c,
    ssize_t stream_array_size
  );

  void seqScale(
    double *a, double *b, double *c,
    ssize_t stream_array_size, double scalar
  );

  void seqAdd(
    double *a, double *b, double *c,
    ssize_t stream_array_size
  );

  void seqTriad(
    double *a, double *b, double *c,
    ssize_t stream_array_size, double scalar
  );

  void gatherCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size
  );

  void gatherScale(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size, double scalar
  );

  void gatherAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t stream_array_size
  );

  void gatherTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t stream_array_size, double scalar
  );

  void scatterCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size
  );

  void scatterScale(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size, double scalar
  );

  void scatterAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size
  );

  void scatterTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size, double scalar
  );

  void sgCopy(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t stream_array_size
  );

  void sgScale(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2,
    ssize_t stream_array_size, double scalar
  );

  void sgAdd(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t stream_array_size
  );

  void sgTriad(
    double *a, double *b, double *c,
    ssize_t *IDX1, ssize_t *IDX2, ssize_t *IDX3,
    ssize_t stream_array_size, double scalar
  );

  void centralCopy(
    double *a, double *b, double *c,
    ssize_t stream_array_size
  );

  void centralScale(
    double *a, double *b, double *c,
    ssize_t stream_array_size,
    double scalar
  );

  void centralAdd(
    double *a, double *b, double *c,
    ssize_t stream_array_size
  );

  void centralTriad(
    double *a, double *b, double *c,
    ssize_t stream_array_size, double scalar
  );
}

#endif // _RS_OMP_H_
#endif // _ENABLE_OMP_