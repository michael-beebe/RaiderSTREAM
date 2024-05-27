//

// _RS_MPI_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_MPI_OMP.h"

#ifdef _RS_MPI_OMP_H_

RS_MPI_OMP::RS_MPI_OMP(const RSOpts& opts) :
  RSBaseImpl("RS_MPI_OMP", opts.getKernelTypeFromName(opts.getKernelName())),
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

RS_MPI_OMP::~RS_MPI_OMP() {}

bool RS_MPI_OMP::allocateData() {
  int myRank  = -1;    /* MPI rank */
  int size    = -1;    /* MPI size (number of PEs) */

  if ( numPEs == 0 ) {
    std::cout << "RS_MPI_OMP::allocateData() - ERROR: 'pes' cannot be 0" << std::endl;
    return false;
  }

  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Calculate the chunk size for each rank */
  ssize_t chunkSize  = streamArraySize / size;
  ssize_t remainder   = streamArraySize % size;

  /* Adjust the chunk size for the last process */
  if ( myRank == size - 1 ) {
    chunkSize += remainder;
  }

  /* Allocate memory for the local chunks in local heap space */
  MPI_Alloc_mem(chunkSize * sizeof(double), MPI_INFO_NULL, &a);
  MPI_Alloc_mem(chunkSize * sizeof(double), MPI_INFO_NULL, &b);
  MPI_Alloc_mem(chunkSize * sizeof(double), MPI_INFO_NULL, &c);
  MPI_Alloc_mem(chunkSize * sizeof(ssize_t), MPI_INFO_NULL, &idx1);
  MPI_Alloc_mem(chunkSize * sizeof(ssize_t), MPI_INFO_NULL, &idx2);
  MPI_Alloc_mem(chunkSize * sizeof(ssize_t), MPI_INFO_NULL, &idx3);

  /* Initialize the local chunks */
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

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool RS_MPI_OMP::freeData() {
  if ( a )    { MPI_Free_mem( a );    }
  if ( b )    { MPI_Free_mem( b );    }
  if ( c )    { MPI_Free_mem( c );    }
  if ( idx1 ) { MPI_Free_mem( idx1 ); }
  if ( idx2 ) { MPI_Free_mem( idx2 ); }
  if ( idx3 ) { MPI_Free_mem( idx3 ); }
  return true;
}

bool RS_MPI_OMP::execute(
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

  int myRank  = -1;    /* MPI rank */
  int size    = -1;    /* MPI size (number of PEs) */
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Barrier(MPI_COMM_WORLD);

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
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqCopy(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SEQ_SCALE:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqScale(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SEQ_ADD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqAdd(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SEQ_TRIAD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqTriad(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    /* GATHER KERNELS */
    case RSBaseImpl::RS_GATHER_COPY:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherCopy(a, b, c, idx1, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_GATHER_SCALE:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherScale(a, b, c, idx1, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_GATHER_ADD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherAdd(a, b, c, idx1, idx2, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_GATHER_TRIAD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherTriad(a, b, c, idx1, idx2, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;
    
    /* SCATTER KERNELS */
    case RSBaseImpl::RS_SCATTER_COPY:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterCopy(a, b, c, idx1, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SCATTER_SCALE:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterScale(a, b, c, idx1, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SCATTER_ADD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterAdd(a, b, c, idx1, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SCATTER_TRIAD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterTriad(a, b, c, idx1, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;
    
    /* SCATTER-GATHER KERNELS */
    case RSBaseImpl::RS_SG_COPY:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgCopy(a, b, c, idx1, idx2, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SG_SCALE:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgScale(a, b, c, idx1, idx2, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SG_ADD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgAdd(a, b, c, idx1, idx2, idx3, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_SG_TRIAD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgTriad(a, b, c, idx1, idx2, idx3, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;
    
    /* CENTRAL KERNELS */
    case RSBaseImpl::RS_CENTRAL_COPY:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralCopy(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_CENTRAL_SCALE:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralScale(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_CENTRAL_ADD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralAdd(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    case RSBaseImpl::RS_CENTRAL_TRIAD:
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralTriad(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;


    /* ALL KERNELS */
    case RSBaseImpl::RS_ALL:
      /* RS_SEQ_COPY */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqCopy(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SEQ_SCALE */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqScale(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SEQ_ADD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqAdd(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SEQ_TRIAD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      seqTriad(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SEQ_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SEQ_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SEQ_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_GATHER_COPY */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherCopy(a, b, c, idx1, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_GATHER_SCALE */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherScale(a, b, c, idx1, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_GATHER_ADD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherAdd(a, b, c, idx1, idx2, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_GATHER_TRIAD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      gatherTriad(a, b, c, idx1, idx2, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_GATHER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_GATHER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_GATHER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SCATTER_COPY */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterCopy(a, b, c, idx1, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SCATTER_SCALE */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterScale(a, b, c, idx1, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SCATTER_ADD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterAdd(a, b, c, idx1, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SCATTER_TRIAD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      scatterTriad(a, b, c, idx1, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SCATTER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SCATTER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SCATTER_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SG_COPY */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgCopy(a, b, c, idx1, idx2, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SG_SCALE */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgScale(a, b, c, idx1, idx2, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SG_ADD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgAdd(a, b, c, idx1, idx2, idx3, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_SG_TRIAD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      sgTriad(a, b, c, idx1, idx2, idx3, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_SG_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_SG_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_SG_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_CENTRAL_COPY */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralCopy(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_COPY], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_CENTRAL_SCALE */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralScale(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_SCALE], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_CENTRAL_ADD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralAdd(a, b, c, chunkSize);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_ADD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* RS_CENTRAL_TRIAD */
      MPI_Barrier(MPI_COMM_WORLD);
      startTime = MPI_Wtime();
      centralTriad(a, b, c, chunkSize, scalar);
      MPI_Barrier(MPI_COMM_WORLD);
      endTime = MPI_Wtime();

      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      
      localRunTime = runTime;
      localMbps = mbps;
      localFlops = flops;

      MPI_Reduce(&localRunTime, &TIMES[RSBaseImpl::RS_CENTRAL_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localMbps, &MBPS[RSBaseImpl::RS_CENTRAL_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&localFlops, &FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      break;

    /* NO KERNELS, SOMETHING IS WRONG */
    default:
      if ( myRank == 0 ) {
        std::cout << "RS_MPI_OMP::execute() - ERROR: KERNEL NOT SET" << std::endl;
      }
      return false;
  }
  return true;
}


#endif /* _RS_MPI_OMP_H_ */

/* EOF */

