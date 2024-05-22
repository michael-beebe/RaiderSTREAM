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

RS_OMP::RS_OMP(const RSOpts& opts) :
  RSBaseImpl("RS_OMP", opts.getKernelTypeFromName(opts.getKernelName())),
  kernelName(opts.getKernelName()),
  streamArraySize(opts.getStreamArraySize()),
  numPEs(opts.getNumPEs()),
  lArgc(0),
  lArgv(nullptr),
  a(nullptr),
  b(nullptr),
  idx1(nullptr),
  idx2(nullptr),
  idx3(nullptr),
  scalar(3.0)
{}

RS_OMP::~RS_OMP() {}

bool RS_OMP::allocateData() {
  a =    new  double[streamArraySize];
  b =    new  double[streamArraySize];
  c =    new  double[streamArraySize];
  idx1 = new ssize_t[streamArraySize];
  idx2 = new ssize_t[streamArraySize];
  idx3 = new ssize_t[streamArraySize];

  initStreamArray(a, streamArraySize, 1.0);
  this->initStreamArray(b, streamArraySize, 2.0);
  this->initStreamArray(c, streamArraySize, 0.0);

  this->initRandomIdxArray(idx1, streamArraySize);
  this->initRandomIdxArray(idx2, streamArraySize);
  this->initRandomIdxArray(idx3, streamArraySize);

  #ifdef _DEBUG_
  std::cout << "===================================================================================" << std::endl;
  std::cout << " RaiderSTREAM Array Info:" << std::endl;
  std::cout << "===================================================================================" << std::endl;
  std::cout << "streamArraySize         = " << streamArraySize << std::endl;
  std::cout << "a[streamArraySize-1]    = " << a[streamArraySize-1] << std::endl;
  std::cout << "b[streamArraySize-1]    = " << b[streamArraySize-1] << std::endl;
  std::cout << "c[streamArraySize-1]    = " << c[streamArraySize-1] << std::endl;
  std::cout << "idx1[streamArraySize-1] = " << idx1[streamArraySize-1] << std::endl;
  std::cout << "idx2[streamArraySize-1] = " << idx2[streamArraySize-1] << std::endl;
  std::cout << "idx3[streamArraySize-1] = " << idx3[streamArraySize-1] << std::endl;
  std::cout << "===================================================================================" << std::endl;
  #endif

  return true;
}

bool RS_OMP::freeData() {
  if ( a ) { delete[] a; }
  if ( b ) { delete[] b; }  
  if ( c ) { delete[] c; }
  if ( idx1 ) { delete[] idx1; }
  if ( idx2 ) { delete[] idx2; }
  if ( idx3 ) { delete[] idx3; }
  return true;
}

bool RS_OMP::execute(
  double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
) {
  double startTime  = 0.0;
  double endTime    = 0.0;
  double runTime    = 0.0;
  double mbps       = 0.0;
  double flops      = 0.0;

  RSBaseImpl::RSKernelType kType = this->getKernelType();

  switch ( kType ) {
    /* SEQUENTIAL KERNELS */
    case RSBaseImpl::RS_SEQ_COPY:
      startTime = this->mySecond();
      seqCopy(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;
      break;

    case RSBaseImpl::RS_SEQ_SCALE:
      startTime = this->mySecond();
      seqScale(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SEQ_ADD:
      startTime = this->mySecond();
      seqAdd(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;
      break;

    case RSBaseImpl::RS_SEQ_TRIAD:
      startTime = this->mySecond();
      seqTriad(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;
      break;

    /* GATHER KERNELS */
    case RSBaseImpl::RS_GATHER_COPY:
      startTime = this->mySecond();
      gatherCopy(a, b, c, idx1, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;
      break;

    case RSBaseImpl::RS_GATHER_SCALE:
      startTime = this->mySecond();
      gatherScale(a, b, c, idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;
      break;

    case RSBaseImpl::RS_GATHER_ADD:
      startTime = this->mySecond();
      gatherAdd(a, b, c, idx1, idx2, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;
      break;

    case RSBaseImpl::RS_GATHER_TRIAD:
      startTime = this->mySecond();
      gatherTriad(a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;
      break;
    
    /* SCATTER KERNELS */
    case RSBaseImpl::RS_SCATTER_COPY:
      startTime = this->mySecond();
      scatterCopy(a, b, c, idx1, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_SCALE:
      startTime = this->mySecond();
      scatterScale(a, b, c, idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_ADD:
      startTime = this->mySecond();
      scatterAdd(a, b, c, idx1, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_TRIAD:
      startTime = this->mySecond();
      scatterTriad(a, b, c, idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;
      break;
    
    /* SCATTER-GATHER KERNELS */
    case RSBaseImpl::RS_SG_COPY:
      startTime = this->mySecond();
      sgCopy(a, b, c, idx1, idx2, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;
      break;

    case RSBaseImpl::RS_SG_SCALE:
      startTime = this->mySecond();
      sgScale(a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SG_ADD:
      startTime = this->mySecond();
      sgAdd(a, b, c, idx1, idx2, idx3, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;
      break;

    case RSBaseImpl::RS_SG_TRIAD:
      startTime = this->mySecond();
      sgTriad(a, b, c, idx1, idx2, idx3, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;
      break;
    
    /* CENTRAL KERNELS */
    case RSBaseImpl::RS_CENTRAL_COPY:
      startTime = this->mySecond();
      centralCopy(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;
      break;

    case RSBaseImpl::RS_CENTRAL_SCALE:
      startTime = this->mySecond();
      centralScale(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;   
      break;

    case RSBaseImpl::RS_CENTRAL_ADD:
      startTime = this->mySecond();
      centralAdd(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;   
      break;

    case RSBaseImpl::RS_CENTRAL_TRIAD:
      startTime = this->mySecond();
      centralTriad(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;   
      break;

    /* ALL KERNELS */
    case RSBaseImpl::RS_ALL:
      /* RS_SEQ_COPY */
      startTime = this->mySecond();
      seqCopy(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;

      /* RS_SEQ_SCALE */
      startTime = this->mySecond();
      seqScale(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;

      /* RS_SEQ_ADD */
      startTime = this->mySecond();
      seqAdd(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;

      /* RS_SEQ_TRIAD */
      startTime = this->mySecond();
      seqTriad(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;

      /* RS_GATHER_COPY */
      startTime = this->mySecond();
      gatherCopy(a, b, c, idx1, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;

      /* RS_GATHER_SCALE */
      startTime = this->mySecond();
      gatherScale(a, b, c, idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;

      /* RS_GATHER_ADD */
      startTime = this->mySecond();
      gatherAdd(a, b, c, idx1, idx2, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;

      /* RS_GATHER_TRIAD */
      startTime = this->mySecond();
      gatherTriad(a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;

      /* RS_SCATTER_COPY */
      startTime = this->mySecond();
      scatterCopy(a, b, c, idx1, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;

      /* RS_SCATTER_SCALE */
      startTime = this->mySecond();
      scatterScale(a, b, c, idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;

      /* RS_SCATTER_ADD */
      startTime = this->mySecond();
      scatterAdd(a, b, c, idx1, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;

      /* RS_SCATTER_TRIAD */
      startTime = this->mySecond();
      scatterTriad(a, b, c, idx1, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;

      /* RS_SG_COPY */
      startTime = this->mySecond();
      sgCopy(a, b, c, idx1, idx2, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;

      /* RS_SG_SCALE */
      startTime = this->mySecond();
      sgScale(a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;

      /* RS_SG_ADD */
      startTime = this->mySecond();
      sgAdd(a, b, c, idx1, idx2, idx3, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;

      /* RS_SG_TRIAD */
      startTime = this->mySecond();
      sgTriad(a, b, c, idx1, idx2, idx3, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;

      /* RS_CENTRAL_COPY */
      startTime = this->mySecond();
      centralCopy(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;

      /* RS_CENTRAL_SCALE */
      startTime = this->mySecond();
      centralScale(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;   

      /* RS_CENTRAL_ADD */
      startTime = this->mySecond();
      centralAdd(a, b, c, streamArraySize);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;   

      /* RS_CENTRAL_TRIAD */
      startTime = this->mySecond();
      centralTriad(a, b, c, streamArraySize, scalar);
      endTime = this->mySecond();
      runTime = this->calculateRunTime(startTime, endTime);
      mbps = this->calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = this->calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops; 
      break;

    /* NO KERNELS, SOMETHING IS WRONG */
    default:
      std::cout << "RS_OMP::execute() - ERROR: KERNEL NOT SET" << std::endl;
      return false;
  }
  return true;
}

#endif /* _RS_OMP_H_ */
/* EOF */

