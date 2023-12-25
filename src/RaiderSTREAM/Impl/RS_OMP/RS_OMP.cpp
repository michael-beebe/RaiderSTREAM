//
// _RS_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_OMP.h"

#ifdef _RS_OMP_H_

#include <sys/types.h>

// RaiderSTREAM OpenMP implementation class constructor
RS_OMP::RS_OMP(
  std::string implName,
  RSBaseImpl::RSKernelType kType
) : 
  RSBaseImpl("RS_OMP", kType),
  kernelName("all"),
  streamArraySize(1000000),
  numTimes(10),
  streamType("double"),
  numPEs(1),
  lArgc(0),
  lArgv(nullptr),
  a(nullptr),
  b(nullptr),
  c(nullptr),
  idx1(nullptr),
  idx2(nullptr),
  idx3(nullptr),
  mbps(nullptr),
  flops(nullptr),
  times(nullptr),
  scalar(3.0)
{}

RS_OMP::~RS_OMP() {}

/*

*/
bool RS_OMP::allocateData(
    double *a, double *b, double *c,
    ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
    double *mbps, double *flops, double *times
) {
  a = (double *) malloc(streamArraySize * sizeof(double));
  b = (double *) malloc(streamArraySize * sizeof(double));
  c = (double *) malloc(streamArraySize * sizeof(double));
  idx1 = (ssize_t *) malloc(streamArraySize * sizeof(ssize_t));
  idx2 = (ssize_t *) malloc(streamArraySize * sizeof(ssize_t));
  idx3 = (ssize_t *) malloc(streamArraySize * sizeof(ssize_t));

  mbps = (double *) malloc(NUM_KERNELS * sizeof(double));
  flops = (double *) malloc(NUM_KERNELS * sizeof(double));
  times = (double *) malloc(NUM_KERNELS * sizeof(double));

  initStreamArray(a, streamArraySize, 1.0);
  initStreamArray(b, streamArraySize, 2.0);
  initStreamArray(c, streamArraySize, 0.0);

  return true;
}

/*

*/
bool RS_OMP::execute(double *times, double *mbps, double *flops) {  
  RSBaseImpl::RSKernelType kType = this->getKernelType();
  double startTime = 0.0;
  double endTime = 0.0;
  double MBPS = 0.0;
  double FLOPS = 0.0;

  // this->allocateData(a, b, c, idx1, idx2, idx3, );

  return true;
}

/*

*/
bool RS_OMP::freeData() {
  return true;
}

#endif // _RS_OMP_H_
// EOF

