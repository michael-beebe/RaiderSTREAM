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
  scalar(3.0)
{}

RS_OMP::~RS_OMP() {}

bool RS_OMP::allocateData(
    double* a, double* b, double* c,
    ssize_t* idx1, ssize_t* idx2, ssize_t* idx3
) {
  this->a = (double *) malloc(streamArraySize * sizeof(double));
  this->b = (double *) malloc(streamArraySize * sizeof(double));
  this->c = (double *) malloc(streamArraySize * sizeof(double));
  this->idx1 = (ssize_t *) malloc(streamArraySize * sizeof(ssize_t));
  this->idx2 = (ssize_t *) malloc(streamArraySize * sizeof(ssize_t));
  this->idx3 = (ssize_t *) malloc(streamArraySize * sizeof(ssize_t));

  initStreamArray(this->a, streamArraySize, 1.0);
  initStreamArray(this->b, streamArraySize, 2.0);
  initStreamArray(this->c, streamArraySize, 0.0);

  a = this->a;
  b = this->b;
  c = this->c;
  idx1 = this->idx1;
  idx2 = this->idx2;
  idx3 = this->idx3;

  return true;
}

bool RS_OMP::execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) {  
  RSBaseImpl::RSKernelType kType = this->getKernelType();
  double startTime = 0.0;
  double endTime = 0.0;
  double runTime = 0.0;
  double mbps = 0.0;
  double flops = 0.0;

  switch ( kType ) {
    /* SEQUENTIAL KERNELS */
    case RSBaseImpl::RS_SEQ_COPY:
      startTime = this->mySecond();
      seqCopy(this->a, this->b, this->c, streamArraySize);
      endTime = this->mySecond();

      runTime = this->calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(MBPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);

      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;

      break;
    case RSBaseImpl::RS_SEQ_SCALE:
      startTime = this->mySecond();
      seqScale(this->a, this->b, this->c, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SEQ_ADD:
      startTime = this->mySecond();
      seqAdd(this->a, this->b, this->c, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SEQ_TRIAD:
      startTime = this->mySecond();
      seqTriad(this->a, this->b, this->c, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;

    /* GATHER KERNELS */
    case RSBaseImpl::RS_GATHER_COPY:
      startTime = this->mySecond();
      gatherCopy(this->a, this->b, this->c, this->idx1, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_GATHER_SCALE:
      startTime = this->mySecond();
      gatherScale(this->a, this->b, this->c, this->idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_GATHER_ADD:
      startTime = this->mySecond();
      gatherAdd(this->a, this->b, this->c, this->idx1, this->idx2, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_GATHER_TRIAD:
      startTime = this->mySecond();
      gatherTriad(this->a, this->b, this->c, this->idx1, this->idx2, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    
    /* SCATTER KERNELS */
    case RSBaseImpl::RS_SCATTER_COPY:
      startTime = this->mySecond();
      scatterCopy(this->a, this->b, this->c, this->idx1, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SCATTER_SCALE:
      startTime = this->mySecond();
      scatterScale(this->a, this->b, this->c, this->idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SCATTER_ADD:
      startTime = this->mySecond();
      scatterAdd(this->a, this->b, this->c, this->idx1, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SCATTER_TRIAD:
      startTime = this->mySecond();
      scatterTriad(this->a, this->b, this->c, this->idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    
    /* SCATTER-GATHER KERNELS */
    case RSBaseImpl::RS_SG_COPY:
      startTime = this->mySecond();
      sgCopy(this->a, this->b, this->c, this->idx1, this->idx2, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SG_SCALE:
      startTime = this->mySecond();
      sgScale(this->a, this->b, this->c, this->idx1, this->idx2, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SG_ADD:
      startTime = this->mySecond();
      sgAdd(this->a, this->b, this->c, this->idx1, this->idx2, this->idx3, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_SG_TRIAD:
      startTime = this->mySecond();
      sgTriad(this->a, this->b, this->c, this->idx1, this->idx2, this->idx3, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    
    /* CENTRAL KERNELS */
    case RSBaseImpl::RS_CENTRAL_COPY:
      startTime = this->mySecond();
      centralCopy(this->a, this->b, this->c, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_CENTRAL_SCALE:
      startTime = this->mySecond();
      centralScale(this->a, this->b, this->c, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_CENTRAL_ADD:
      startTime = this->mySecond();
      centralAdd(this->a, this->b, this->c, streamArraySize);
      endTime = this->mySecond();
      // TODO: record results
      break;
    case RSBaseImpl::RS_CENTRAL_TRIAD:
      startTime = this->mySecond();
      centralTriad(this->a, this->b, this->c, streamArraySize, scalar);
      endTime = this->mySecond();
      // TODO: record results
      break;
    
    /* ALL KERNELS */
    // TODO: case RSBaseImpl::RS_ALL

    /* NO KERNELS, SOMETHING IS WRONG */
    default:
      std::cout << "ERROR: KERNEL NOT SET" << std::endl;
      return false;
  }
  return true;
}

bool RS_OMP::freeData() {
  free(this->a);
  free(this->b);
  free(this->c);
  free(this->idx1);
  free(this->idx2);
  free(this->idx3);

  this->a = nullptr;
  this->b = nullptr;
  this->c = nullptr;
  this->idx1 = nullptr;
  this->idx2 = nullptr;
  this->idx3 = nullptr;

  return true;
}

#endif // _RS_OMP_H_
// EOF

