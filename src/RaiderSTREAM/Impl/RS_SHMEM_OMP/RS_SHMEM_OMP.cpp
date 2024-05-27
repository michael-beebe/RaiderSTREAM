//
// _RS_SHMEM_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_SHMEM_OMP.h"

#ifdef _RS_SHMEM_OMP_H_

RS_SHMEM_OMP::RS_SHMEM_OMP(const RSOpts& opts) :
  RSBaseImpl("RS_SHMEM_OMP", opts.getKernelTypeFromName(opts.getKernelName())),
  kernelName(opts.getKernelName()),
  streamArraySize(opts.getStreamArraySize()),
  lArgc(0),
  lArgv(nullptr),
  numPEs(opts.getNumPEs()),
  a(nullptr),
  b(nullptr),
  idx1(nullptr),
  idx2(nullptr),
  idx3(nullptr),
  scalar(3.0)
{}

RS_SHMEM_OMP::~RS_SHMEM_OMP() {}

bool RS_SHMEM_OMP::allocateData() {
  int myRank  = shmem_my_pe(); /* Current rank */
  int size    = shmem_n_pes(); /* Number of shmem ranks */

  if ( numPEs == 0 ) {
    std::cout << "RS_SHMEM_OMP::allocateData() - ERROR: 'pes' cannot be 0" << std::endl;
    return false;
  }

  shmem_barrier_all();

  /* Calculate the chunk size for each rank */
  ssize_t chunkSize  = streamArraySize / size;
  ssize_t remainder   = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if ( myRank == size - 1 ) {
    chunkSize += remainder;
  }

  /* Allocate memory for the local chunks in symmetric heap space */
  a     =  static_cast<double*>(shmem_malloc(chunkSize * sizeof(double)));
  b     =  static_cast<double*>(shmem_malloc(chunkSize * sizeof(double)));
  c     =  static_cast<double*>(shmem_malloc(chunkSize * sizeof(double)));
  idx1  = static_cast<ssize_t*>(shmem_malloc(chunkSize * sizeof(ssize_t)));
  idx2  = static_cast<ssize_t*>(shmem_malloc(chunkSize * sizeof(ssize_t)));
  idx3  = static_cast<ssize_t*>(shmem_malloc(chunkSize * sizeof(ssize_t)));

  /* Initialize the local chunks */
  initStreamArray(a, chunkSize, 1.0);
  initStreamArray(b, chunkSize, 2.0);
  initStreamArray(c, chunkSize, 0.0);

  #ifdef _ARRAYGEN_
    initReadIdxArray(idx1, chunkSize, "RaiderSTREAM/arraygen/IDX1.txt");
    initReadIdxArray(idx2, chunkSize, "RaiderSTREAM/arraygen/IDX2.txt");
    initReadIdxArray(idx3, chunkSize, "RaiderSTREAM/arraygen/IDX3.txt");
  #else
    initRandomIdxArray(idx1, chunkSize);
    initRandomIdxArray(idx2, chunkSize);
    initRandomIdxArray(idx3, chunkSize);
  #endif

  #ifdef _DEBUG_
    if ( myRank == 0 ) {
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
    }
  #endif

  shmem_barrier_all();

  return true;
}

bool RS_SHMEM_OMP::freeData() {
  if ( a )    { shmem_free( a );    }
  if ( b )    { shmem_free( b );    }
  if ( c )    { shmem_free( c );    }
  if ( idx1 ) { shmem_free( idx1 ); }
  if ( idx2 ) { shmem_free( idx2 ); }
  if ( idx3 ) { shmem_free( idx3 ); }
  return true;
}

bool RS_SHMEM_OMP::execute(
  double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
) {
  double startTime  = 0.0;
  double endTime    = 0.0;
  double runTime    = 0.0;
  double mbps       = 0.0;
  double flops      = 0.0;

  double localRunTime = 0.0;
  double localMbps    = 0.0;
  double localFlops   = 0.0;

  int myRank  = shmem_my_pe(); /* Current rank */
  int size    = shmem_n_pes(); /* Number of shmem ranks */
  size_t syncSize = SHMEM_SYNC_SIZE;

  double *pWrk = static_cast<double*>(shmem_malloc(size * sizeof(double)));
  long *pSync = static_cast<long*>(shmem_malloc(syncSize * sizeof(long)));
  for (size_t i = 0; i < syncSize; ++i) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }

  shmem_barrier_all();

  /* Calculate the chunk size for each rank */
  ssize_t chunkSize  = streamArraySize / size;
  ssize_t remainder   = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if ( myRank == size - 1 ) {
    chunkSize += remainder;
  }

  RSBaseImpl::RSKernelType kType = getKernelType();

  switch ( kType ) {
    /* SEQUENTIAL KERNELS */
    case RSBaseImpl::RS_SEQ_COPY:
      shmem_barrier_all();
      startTime = mySecond();
      seqCopy(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);

      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SEQ_SCALE:
      shmem_barrier_all();
      startTime = mySecond();
      seqScale(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SEQ_ADD:
      shmem_barrier_all();
      startTime = mySecond();
      seqAdd(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SEQ_TRIAD:
      shmem_barrier_all();
      startTime = mySecond();
      seqTriad(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    /* GATHER KERNELS */
    case RSBaseImpl::RS_GATHER_COPY:
      shmem_barrier_all();
      startTime = mySecond();
      gatherCopy(a, b, c, idx1, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_GATHER_SCALE:
      shmem_barrier_all();
      startTime = mySecond();
      gatherScale(a, b, c, idx1, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_GATHER_ADD:
      shmem_barrier_all();
      startTime = mySecond();
      gatherAdd(a, b, c, idx1, idx2, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_GATHER_TRIAD:
      shmem_barrier_all();
      startTime = mySecond();
      gatherTriad(a, b, c, idx1, idx2, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;
    
    /* SCATTER KERNELS */
    case RSBaseImpl::RS_SCATTER_COPY:
      shmem_barrier_all();
      startTime = mySecond();
      scatterCopy(a, b, c, idx1, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SCATTER_SCALE:
      shmem_barrier_all();
      startTime = mySecond();
      scatterScale(a, b, c, idx1, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SCATTER_ADD:
      shmem_barrier_all();
      startTime = mySecond();
      scatterAdd(a, b, c, idx1, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SCATTER_TRIAD:
      shmem_barrier_all();
      startTime = mySecond();
      scatterTriad(a, b, c, idx1, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;
    
    /* SCATTER-GATHER KERNELS */
    case RSBaseImpl::RS_SG_COPY:
      shmem_barrier_all();
      startTime = mySecond();
      sgCopy(a, b, c, idx1, idx2, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SG_SCALE:
      shmem_barrier_all();
      startTime = mySecond();
      sgScale(a, b, c, idx1, idx2, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SG_ADD:
      shmem_barrier_all();
      startTime = mySecond();
      sgAdd(a, b, c, idx1, idx2, idx3, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_SG_TRIAD:
      shmem_barrier_all();
      startTime = mySecond();
      sgTriad(a, b, c, idx1, idx2, idx3, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;
    
    /* CENTRAL KERNELS */
    case RSBaseImpl::RS_CENTRAL_COPY:
      shmem_barrier_all();
      startTime = mySecond();
      centralCopy(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_CENTRAL_SCALE:
      shmem_barrier_all();
      startTime = mySecond();
      centralScale(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_CENTRAL_ADD:
      shmem_barrier_all();
      startTime = mySecond();
      centralAdd(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

    case RSBaseImpl::RS_CENTRAL_TRIAD:
      shmem_barrier_all();
      startTime = mySecond();
      centralTriad(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);
      break;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* ALL KERNELS */
    case RSBaseImpl::RS_ALL:
      /* RS_SEQ_COPY */
      shmem_barrier_all();
      startTime = mySecond();
      seqCopy(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SEQ_SCALE */
      shmem_barrier_all();
      startTime = mySecond();
      seqScale(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SEQ_ADD */
      shmem_barrier_all();
      startTime = mySecond();
      seqAdd(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SEQ_TRIAD */
      shmem_barrier_all();
      startTime = mySecond();
      seqTriad(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SEQ_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SEQ_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SEQ_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_GATHER_COPY */
      shmem_barrier_all();
      startTime = mySecond();
      gatherCopy(a, b, c, idx1, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_GATHER_SCALE */
      shmem_barrier_all();
      startTime = mySecond();
      gatherScale(a, b, c, idx1, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_GATHER_ADD */
      shmem_barrier_all();
      startTime = mySecond();
      gatherAdd(a, b, c, idx1, idx2, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_GATHER_TRIAD */
      shmem_barrier_all();
      startTime = mySecond();
      gatherTriad(a, b, c, idx1, idx2, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_GATHER_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_GATHER_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_GATHER_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SCATTER_COPY */
      shmem_barrier_all();
      startTime = mySecond();
      scatterCopy(a, b, c, idx1, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SCATTER_SCALE */
      shmem_barrier_all();
      startTime = mySecond();
      scatterScale(a, b, c, idx1, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SCATTER_ADD */
      shmem_barrier_all();
      startTime = mySecond();
      scatterAdd(a, b, c, idx1, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SCATTER_TRIAD */
      shmem_barrier_all();
      startTime = mySecond();
      scatterTriad(a, b, c, idx1, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SCATTER_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SCATTER_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SCATTER_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SG_COPY */
      shmem_barrier_all();
      startTime = mySecond();
      sgCopy(a, b, c, idx1, idx2, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SG_SCALE */
      shmem_barrier_all();
      startTime = mySecond();
      sgScale(a, b, c, idx1, idx2, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SG_ADD */
      shmem_barrier_all();
      startTime = mySecond();
      sgAdd(a, b, c, idx1, idx2, idx3, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_SG_TRIAD */
      shmem_barrier_all();
      startTime = mySecond();
      sgTriad(a, b, c, idx1, idx2, idx3, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_SG_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_SG_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_SG_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_CENTRAL_COPY */
      shmem_barrier_all();
      startTime = mySecond();
      centralCopy(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_COPY], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_COPY], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_COPY], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_CENTRAL_SCALE */
      shmem_barrier_all();
      startTime = mySecond();
      centralScale(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_SCALE], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_SCALE], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_SCALE], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_CENTRAL_ADD */
      shmem_barrier_all();
      startTime = mySecond();
      centralAdd(a, b, c, chunkSize);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_ADD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_ADD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_ADD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      shmem_free(pWrk);
      shmem_free(pSync);

      /* RS_CENTRAL_TRIAD */
      shmem_barrier_all();
      startTime = mySecond();
      centralTriad(a, b, c, chunkSize, scalar);
      shmem_barrier_all();
      endTime = mySecond();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      pWrk = static_cast<double*>(shmem_malloc( size * sizeof(double) ));
      pSync = static_cast<long*>(shmem_malloc( size * sizeof(long) ));

      shmem_double_max_to_all(&TIMES[RSBaseImpl::RS_CENTRAL_TRIAD], &localRunTime, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&MBPS[RSBaseImpl::RS_CENTRAL_TRIAD], &localMbps, 1, 0, 0, size, pWrk, pSync);
      shmem_double_max_to_all(&FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD], &localFlops, 1, 0, 0, size, pWrk, pSync);

      break;

    /* NO KERNELS, SOMETHING IS WRONG */
    default:
      if ( myRank == 0 ) {
        std::cout << "RS_SHMEM_OMP::execute() - ERROR: KERNEL NOT SET" << std::endl;
      }
      return false;
  }

  shmem_free(pWrk);
  shmem_free(pSync);
  return true;
}


#endif /* _RS_SHMEM_OMP_H_ */

/* EOF */

