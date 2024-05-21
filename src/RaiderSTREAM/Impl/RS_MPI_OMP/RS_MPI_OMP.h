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

#include <mpih.h>
#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

class RS_MPI_OMP : public RSBaseImpl {
private:
  std::string kernelNameClass
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
  RS_MPI_OMP(const RSOpts& opts);

  ~RS_OMP();

  virtual bool execute(
    double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
  ) override;

  virtual bool allocateData() override;

  virtual bool freeData() override;
}

extern "C" {
  // TODO:
}

#endif // _RS_MPI_OMP_H_
#endif // _ENABLE_MPI_OMP_
