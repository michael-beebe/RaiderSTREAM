//
// _RS_OACC_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_OACC.h"

#ifdef _RS_OACC_H_
#define _DEBUG_


#include<stdio.h>

RS_OACC::RS_OACC(const RSOpts& opts) :
  RSBaseImpl("RS_OACC", opts.getKernelTypeFromName(opts.getKernelName())),
  kernelName(opts.getKernelName()),
  streamArraySize(opts.getStreamArraySize()),
  numPEs(opts.getNumPEs()),
  numGangs(opts.getThreadBlocks()),
  numWorkers(opts.getThreadsPerBlocks()),
  lArgc(0),
  lArgv(nullptr),
  a(nullptr),
  b(nullptr),
  c(nullptr),
  d_a(nullptr),
  d_b(nullptr),
  d_c(nullptr),
  idx1(nullptr),
  idx2(nullptr),
  idx3(nullptr),
  d_idx1(nullptr),
  d_idx2(nullptr),
  d_idx3(nullptr),
  scalar(3.0)
{}

RS_OACC::~RS_OACC() {}

bool RS_OACC::allocateData() {

  int StreamMemArraySize = streamArraySize * sizeof(double); //Multiply size of the array by the size of the variable in the array to get total memory size
  int IdxMemArraySize = streamArraySize * sizeof(ssize_t); //Multiply size of the array by the size of the variable in the array to get total memory size

  a =    new  double[streamArraySize]; //Create a pointer in host memory for a
  b =    new  double[streamArraySize]; //Create a pointer in host memory for b
  c =    new  double[streamArraySize]; //Create a pointer in host memory for c
  idx1 = new ssize_t[streamArraySize]; //Create a pointer in host memory for idx1
  idx2 = new ssize_t[streamArraySize]; //Create a pointer in host memory for idx2
  idx3 = new ssize_t[streamArraySize]; //Create a pointer in host memory for idx3

  #ifdef _ARRAYGEN_
    initReadIdxArray(idx1, streamArraySize, "RaiderSTREAM/arraygen/IDX1.txt");
    initReadIdxArray(idx2, streamArraySize, "RaiderSTREAM/arraygen/IDX2.txt");
    initReadIdxArray(idx3, streamArraySize, "RaiderSTREAM/arraygen/IDX3.txt");
  #else
    initRandomIdxArray(idx1, streamArraySize);
    initRandomIdxArray(idx2, streamArraySize);
    initRandomIdxArray(idx3, streamArraySize);
  #endif


  /* a -> d_a */
  d_a = (double*) acc_malloc(StreamMemArraySize); //Allocating memory in GPU/device
  acc_memcpy_to_device(d_a, a, StreamMemArraySize); //Copying data from a (on the host) to d_a (on the device)
  std::cout << "Device Pointer is: " << d_a << std::endl;
  std::cout << "Host Pointer is: " << a << std::endl;
    /* b -> d_b */
  d_b = (double*) acc_malloc(StreamMemArraySize);
  acc_memcpy_to_device(d_b, b, StreamMemArraySize);
  std::cout << "Device Pointer is: " << d_b << std::endl;
  std::cout << "Host Pointer is: " << b << std::endl;
    /* c -> d_c */
  d_c = (double*) acc_malloc(StreamMemArraySize);
  acc_memcpy_to_device(d_c, c, StreamMemArraySize);
  std::cout << "Device Pointer is: " << d_c << std::endl;
  std::cout << "Host Pointer is: " << c << std::endl;
    /* idx1 -> d_idx1 */
  d_idx1 = (ssize_t*) acc_malloc(IdxMemArraySize);
  acc_memcpy_to_device(d_idx1, idx1, IdxMemArraySize);
  std::cout << "Device Pointer is: " << d_idx1 << std::endl;
  std::cout << "Host Pointer is: " << idx1 << std::endl;
    /* idx2 -> d_idx2 */
  d_idx2 = (ssize_t*) acc_malloc(IdxMemArraySize);
  acc_memcpy_to_device(d_idx2, idx2, IdxMemArraySize);
  std::cout << "Device Pointer is: " << d_idx2 << std::endl;
  std::cout << "Host Pointer is: " << idx2 << std::endl;
    /* idx3 -> d_idx3 */
  d_idx3 = (ssize_t*) acc_malloc(IdxMemArraySize);
  acc_memcpy_to_device(d_idx3, idx3, IdxMemArraySize);
  std::cout << "Device Pointer is: " << d_idx3 << std::endl;
  std::cout << "Host Pointer is: " << idx3 << std::endl;

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
  double *testarray = new double[streamArraySize];
  acc_memcpy_from_device(testarray, d_a, StreamMemArraySize);
  std::cout << testarray[streamArraySize-1] <<std::endl;
  testarray = new double[streamArraySize];
  acc_memcpy_from_device(testarray, d_b, StreamMemArraySize);
  std::cout << testarray[streamArraySize-1] <<std::endl;
  testarray = new double[streamArraySize];
  acc_memcpy_from_device(testarray, d_c, StreamMemArraySize);
  std::cout << testarray[streamArraySize-1] <<std::endl;
  delete[] testarray;
  ssize_t *test2 = new ssize_t[streamArraySize];
  acc_memcpy_from_device(test2, d_idx1, IdxMemArraySize);
  std::cout << test2[streamArraySize-1] <<std::endl;
  test2 = new ssize_t[streamArraySize];
  acc_memcpy_from_device(test2, d_idx2, IdxMemArraySize);
  std::cout << test2[streamArraySize-1] <<std::endl;
  test2 = new ssize_t[streamArraySize];
  acc_memcpy_from_device(test2, d_idx3, IdxMemArraySize);
  std::cout << test2[streamArraySize-1] <<std::endl;
  delete[] test2;
  std::cout << "===================================================================================" << std::endl;
  #endif

  
  return true;
}

bool RS_OACC::freeData() {
  if ( a ) { delete[] a; }
  if ( b ) { delete[] b; }  
  if ( c ) { delete[] c; }
  if ( idx1 ) { delete[] idx1; }
  if ( idx2 ) { delete[] idx2; }
  if ( idx3 ) { delete[] idx3; }
  if ( d_a ) { acc_free(d_a); }
  if ( d_b ) { acc_free(d_b); }
  if ( d_c ) { acc_free(d_c); }
  if ( d_idx1 ) { acc_free(d_idx1); }
  if ( d_idx2 ) { acc_free(d_idx2); }
  if ( d_idx3 ) { acc_free(d_idx3); }
  return true;
}

bool RS_OACC::execute(
  double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
) {
  double startTime  = 0.0;
  double endTime    = 0.0;
  double runTime    = 0.0;
  double mbps       = 0.0;
  double flops      = 0.0;

  RSBaseImpl::RSKernelType kType = getKernelType();

  switch ( kType ) {
    /* SEQUENTIAL KERNELS */
    case RSBaseImpl::RS_SEQ_COPY:
      startTime = mySecond();
      std::cout << "Device Pointer is: " << d_a << std::endl;
      std::cout << "Host Pointer is: " << a << std::endl;
      std::cout << "WE MADE IT TO THE KERNEL"<< std::endl;
      #pragma acc data deviceptr(d_a, d_b, d_c)
      seqCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
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
      seqScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
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
      seqAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
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
      seqTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
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
      gatherCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize);
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
      gatherScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize, scalar);
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
      gatherAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
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
      gatherTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize, scalar);
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
      scatterCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize);
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
      scatterScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize, scalar);
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
      scatterAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize);
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
      scatterTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize, scalar);
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
      sgCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
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
      sgScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize, scalar);
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
      sgAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, streamArraySize);
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
      sgTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, streamArraySize, scalar);
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
      centralCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
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
      centralScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
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
      centralAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
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
      centralTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
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
      seqCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;

      /* RS_SEQ_SCALE */
      startTime = mySecond();
      seqScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;

      /* RS_SEQ_ADD */
      startTime = mySecond();
      seqAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;

      /* RS_SEQ_TRIAD */
      startTime = mySecond();
      seqTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;

      /* RS_GATHER_COPY */
      startTime = mySecond();
      gatherCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;

      /* RS_GATHER_SCALE */
      startTime = mySecond();
      gatherScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;

      /* RS_GATHER_ADD */
      startTime = mySecond();
      gatherAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;

      /* RS_GATHER_TRIAD */
      startTime = mySecond();
      gatherTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;

      /* RS_SCATTER_COPY */
      startTime = mySecond();
      scatterCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;

      /* RS_SCATTER_SCALE */
      startTime = mySecond();
      scatterScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;

      /* RS_SCATTER_ADD */
      startTime = mySecond();
      scatterAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;

      /* RS_SCATTER_TRIAD */
      startTime = mySecond();
      scatterTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;

      /* RS_SG_COPY */
      startTime = mySecond();
      sgCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;

      /* RS_SG_SCALE */
      startTime = mySecond();
      sgScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;

      /* RS_SG_ADD */
      startTime = mySecond();
      sgAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;

      /* RS_SG_TRIAD */
      startTime = mySecond();
      sgTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, d_idx3, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;

      /* RS_CENTRAL_COPY */
      startTime = mySecond();
      centralCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;

      /* RS_CENTRAL_SCALE */
      startTime = mySecond();
      centralScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;   

      /* RS_CENTRAL_ADD */
      startTime = mySecond();
      centralAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;   

      /* RS_CENTRAL_TRIAD */
      startTime = mySecond();
      centralTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
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
      std::cout << "RS_OACC::execute() - ERROR: KERNEL NOT SET" << std::endl;
      return false;
  }
  return true;
}

#endif /* _RS_OACC_H_ */
/* EOF */

