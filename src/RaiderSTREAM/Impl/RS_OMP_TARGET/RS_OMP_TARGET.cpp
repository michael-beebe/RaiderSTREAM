//
// _RS_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_OMP_TARGET.h"

#ifdef _RS_OMP_TARGET_H_

RS_OMP_TARGET::RS_OMP_TARGET(const RSOpts& opts) :
  RSBaseImpl("RS_OMP_TARGET", opts.getKernelTypeFromName(opts.getKernelName())),
  kernelName(opts.getKernelName()),
  streamArraySize(opts.getStreamArraySize()),
  numPEs(opts.getNumPEs()),
  numTeams(opts.getThreadBlocks()),
  threadsPerTeam(opts.getThreadsPerBlocks()),
  lArgc(0),
  lArgv(nullptr),
  a(nullptr),
  b(nullptr),
  c(nullptr),
  idx1(nullptr),
  idx2(nullptr),
  idx3(nullptr),
  aOnDevice(nullptr),
  bOnDevice(nullptr),
  cOnDevice(nullptr),
  idx1OnDevice(nullptr),
  idx2OnDevice(nullptr),
  idx3OnDevice(nullptr),
  scalar(3.0)
{}

RS_OMP_TARGET::~RS_OMP_TARGET() {}

bool RS_OMP_TARGET::allocateData() {
  a =    new  double[streamArraySize];
  b =    new  double[streamArraySize];
  c =    new  double[streamArraySize];
  idx1 = new ssize_t[streamArraySize];
  idx2 = new ssize_t[streamArraySize];
  idx3 = new ssize_t[streamArraySize];

  #ifdef _ARRAYGEN_
    initReadIdxArray(idx1, streamArraySize, "RaiderSTREAM/arraygen/IDX1.txt");
    initReadIdxArray(idx2, streamArraySize, "RaiderSTREAM/arraygen/IDX2.txt");
    initReadIdxArray(idx3, streamArraySize, "RaiderSTREAM/arraygen/IDX3.txt");
  #else
    initRandomIdxArray(idx1, streamArraySize);
    initRandomIdxArray(idx2, streamArraySize);
    initRandomIdxArray(idx3, streamArraySize);
  #endif

  int device = omp_get_default_device();
  aOnDevice = (double *) omp_target_alloc(streamArraySize * sizeof(double), device);
  bOnDevice = (double *) omp_target_alloc(streamArraySize * sizeof(double), device);
  cOnDevice = (double *) omp_target_alloc(streamArraySize * sizeof(double), device);
  idx1OnDevice = (ssize_t *) omp_target_alloc(streamArraySize * sizeof(ssize_t), device);
  idx2OnDevice = (ssize_t *) omp_target_alloc(streamArraySize * sizeof(ssize_t), device);
  idx3OnDevice = (ssize_t *) omp_target_alloc(streamArraySize * sizeof(ssize_t), device);

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

bool RS_OMP_TARGET::freeData() {
  if ( a ) { delete[] a; }
  if ( b ) { delete[] b; }  
  if ( c ) { delete[] c; }
  if ( idx1 ) { delete[] idx1; }
  if ( idx2 ) { delete[] idx2; }
  if ( idx3 ) { delete[] idx3; }
  int device = omp_get_default_device();
  if ( aOnDevice ) { omp_target_free( aOnDevice, device ); }
  if ( bOnDevice ) { omp_target_free( bOnDevice, device ); }
  if ( cOnDevice ) { omp_target_free( cOnDevice, device ); }
  if ( idx1OnDevice ) { omp_target_free( idx1OnDevice, device ); }
  if ( idx2OnDevice ) { omp_target_free( idx2OnDevice, device ); }
  if ( idx3OnDevice ) { omp_target_free( idx3OnDevice, device ); }
  return true;
}

void RS_OMP_TARGET::prepare() {
  #pragma omp target data map(from: a[0:streamArraySize], b[0:streamArraySize], c[0:streamArraySize])

  // do we even need to bother recopying data?
  #pragma omp target teams distribute parallel for is_device_ptr(aOnDevice, bOnDevice, cOnDevice)
  for(int i = 0; i < streamArraySize; i++) {
    aOnDevice[i] = a[i];
    bOnDevice[i] = b[i];
    cOnDevice[i] = c[i];
  }
}

bool RS_OMP_TARGET::execute(
  double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
) {
  double startTime  = 0.0;
  double endTime    = 0.0;
  double runTime    = 0.0;
  double mbps       = 0.0;
  double flops      = 0.0;

  RSBaseImpl::RSKernelType kType = getKernelType();

  prepare();
  switch ( kType ) {
    /* SEQUENTIAL KERNELS */
    case RSBaseImpl::RS_SEQ_COPY:
      startTime = mySecond();
      seqCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;
      break;

    case RSBaseImpl::RS_SEQ_SCALE:
      startTime = mySecond();
      seqScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SEQ_ADD:
      startTime = mySecond();
      seqAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;
      break;

    case RSBaseImpl::RS_SEQ_TRIAD:
      startTime = mySecond();
      seqTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;
      break;

    /* GATHER KERNELS */
    case RSBaseImpl::RS_GATHER_COPY:
      startTime = mySecond();
      gatherCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;
      break;

    case RSBaseImpl::RS_GATHER_SCALE:
      startTime = mySecond();
      gatherScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;
      break;

    case RSBaseImpl::RS_GATHER_ADD:
      startTime = mySecond();
      gatherAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;
      break;

    case RSBaseImpl::RS_GATHER_TRIAD:
      startTime = mySecond();
      gatherTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;
      break;
    
    /* SCATTER KERNELS */
    case RSBaseImpl::RS_SCATTER_COPY:
      startTime = mySecond();
      scatterCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_SCALE:
      startTime = mySecond();
      scatterScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_ADD:
      startTime = mySecond();
      scatterAdd(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_TRIAD:
      startTime = mySecond();
      scatterTriad(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;
      break;
    
    /* SCATTER-GATHER KERNELS */
    case RSBaseImpl::RS_SG_COPY:
      startTime = mySecond();
      sgCopy(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;
      break;

    case RSBaseImpl::RS_SG_SCALE:
      startTime = mySecond();
      sgScale(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SG_ADD:
      startTime = mySecond();
      sgAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;
      break;

    case RSBaseImpl::RS_SG_TRIAD:
      startTime = mySecond();
      sgTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;
      break;
    
    /* CENTRAL KERNELS */
    case RSBaseImpl::RS_CENTRAL_COPY:
      startTime = mySecond();
      centralCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;
      break;

    case RSBaseImpl::RS_CENTRAL_SCALE:
      startTime = mySecond();
      centralScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;   
      break;

    case RSBaseImpl::RS_CENTRAL_ADD:
      startTime = mySecond();
      centralAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;   
      break;

    case RSBaseImpl::RS_CENTRAL_TRIAD:
      startTime = mySecond();
      centralTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;   
      break;

    /* ALL KERNELS */
    case RSBaseImpl::RS_ALL:
      /* RS_SEQ_COPY */
      startTime = mySecond();
      seqCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;

      /* RS_SEQ_SCALE */
      startTime = mySecond();
      seqScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;

      /* RS_SEQ_ADD */
      startTime = mySecond();
      seqAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;

      /* RS_SEQ_TRIAD */
      startTime = mySecond();
      seqTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;

      /* RS_GATHER_COPY */
      startTime = mySecond();
      gatherCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;

      /* RS_GATHER_SCALE */
      startTime = mySecond();
      gatherScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;

      /* RS_GATHER_ADD */
      startTime = mySecond();
      gatherAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;

      /* RS_GATHER_TRIAD */
      startTime = mySecond();
      gatherTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;

      /* RS_SCATTER_COPY */
      startTime = mySecond();
      scatterCopy(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;

      /* RS_SCATTER_SCALE */
      startTime = mySecond();
      scatterScale(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;

      /* RS_SCATTER_ADD */
      startTime = mySecond();
      scatterAdd(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;

      /* RS_SCATTER_TRIAD */
      startTime = mySecond();
      scatterTriad(numTeams, threadsPerTeam, a, b, c, idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;

      /* RS_SG_COPY */
      startTime = mySecond();
      sgCopy(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;

      /* RS_SG_SCALE */
      startTime = mySecond();
      sgScale(numTeams, threadsPerTeam, a, b, c, idx1, idx2, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;

      /* RS_SG_ADD */
      startTime = mySecond();
      sgAdd(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;

      /* RS_SG_TRIAD */
      startTime = mySecond();
      sgTriad(numTeams, threadsPerTeam, a, b, c, idx1, idx2, idx3, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;

      /* RS_CENTRAL_COPY */
      startTime = mySecond();
      centralCopy(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;

      /* RS_CENTRAL_SCALE */
      startTime = mySecond();
      centralScale(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;   

      /* RS_CENTRAL_ADD */
      startTime = mySecond();
      centralAdd(numTeams, threadsPerTeam, a, b, c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;   

      /* RS_CENTRAL_TRIAD */
      startTime = mySecond();
      centralTriad(numTeams, threadsPerTeam, a, b, c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops; 
      break;

    /* NO KERNELS, SOMETHING IS WRONG */
    default:
      std::cout << "RS_OMP_TARGET::execute() - ERROR: KERNEL NOT SET" << std::endl;
      return false;
  }
  return true;
}

#endif /* _RS_OMP_TARGET_H_ */
/* EOF */

