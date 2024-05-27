//
// _RS_CUDA_CU_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifdef _ENABLE_CUDA_

#include "RS_CUDA.cuh"

RS_CUDA::RS_CUDA(const RSOpts& opts) :
  RSBaseImpl("RS_OMP", opts.getKernelTypeFromName(opts.getKernelName())),
  kernelName(opts.getKernelName()),
  streamArraySize(opts.getStreamArraySize()),
  numPEs(opts.getNumPEs()),
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
  scalar(3.0),
  threadBlocks(opts.getThreadBlocks()),
  threadsPerBlock(opts.getThreadsPerBlocks())
{}

RS_CUDA::~RS_CUDA() {}

bool RS_CUDA::allocateData() {
  if ( threadBlocks <= 0 ) {
    std::cout << "RS_CUDA::AllocateData: threadBlocks must be greater than 0" << std::endl;
    return false;
  }
  if ( threadsPerBlock <= 0 ) {
      std::cout << "RS_CUDA::AllocateData: threadsPerBlock must be greater than 0" << std::endl;
      return false;
  }

  /* Allocate host memory */
  a =    new  double[streamArraySize];
  b =    new  double[streamArraySize];
  c =    new  double[streamArraySize];
  idx1 = new ssize_t[streamArraySize];
  idx2 = new ssize_t[streamArraySize];
  idx3 = new ssize_t[streamArraySize];

  streamArrayMemSize  = streamArraySize * sizeof(double);
  idxArrayMemSize     = streamArraySize * sizeof(ssize_t);

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
  if(cudaMalloc( &d_a, streamArrayMemSize ) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_a' could not be allocated on device" << std::endl;
    cudaFree(d_a);
    free(a); free(b); free(c);
    free(idx1); free(idx2); free(idx3);
    return false;
  }
  if ( cudaMemcpy(d_a, a, streamArrayMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_a' could not be copied to device" << std::endl;
    return false;
  }

  /* b -> d_b */
  if(cudaMalloc( &d_b, streamArrayMemSize ) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_b' could not be allocated on device" << std::endl;
    cudaFree(d_b);
    free(a); free(b); free(c);
    free(idx1); free(idx2); free(idx3);
    return false;
  }
  if ( cudaMemcpy(d_b, b, streamArrayMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_b' could not be copied to device" << std::endl;
    return false;
  }

  /* c -> d_c */
  if(cudaMalloc( &d_c, streamArrayMemSize ) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_c' could not be allocated on device" << std::endl;
    cudaFree(d_c);
    free(a); free(b); free(c);
    free(idx1); free(idx2); free(idx3);
    return false;
  }
  if ( cudaMemcpy(d_c, c, streamArrayMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_c' could not be copied to device" << std::endl;
    return false;
  }

  /* idx1 -> d_idx1 */
  if(cudaMalloc( &d_idx1, idxArrayMemSize ) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_idx1' could not be allocated on device" << std::endl;
    cudaFree(d_idx1);
    free(a); free(b); free(c);
    free(idx1); free(idx2); free(idx3);
    return false;
  }
  if ( cudaMemcpy(d_idx1, idx1, idxArrayMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_idx1' could not be copied to device" << std::endl;
    return false;
  }

  /* idx2 -> d_idx2 */ 
  if(cudaMalloc( &d_idx2, idxArrayMemSize ) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_idx2' could not be allocated on device" << std::endl;
    cudaFree(d_idx2);
    free(a); free(b); free(c);
    free(idx1); free(idx2); free(idx3);
    return false;
  }
  if ( cudaMemcpy(d_idx2, idx2, idxArrayMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_idx2' could not be copied to device" << std::endl;
    return false;
  }

  /* idx3 -> d_idx3 */ 
  if(cudaMalloc( &d_idx3, idxArrayMemSize ) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_idx3' could not be allocated on device" << std::endl;
    cudaFree(d_idx3);
    free(a); free(b); free(c);
    free(idx1); free(idx2); free(idx3);
    return false;
  }
  if ( cudaMemcpy(d_idx3, idx3, idxArrayMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
    std::cout << "RS_CUDA::AllocateData : 'd_idx3' could not be copied to device" << std::endl;
    return false;
  }

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

bool RS_CUDA::freeData() {
  if ( a ) { delete[] a; }
  if ( b ) { delete[] b; }  
  if ( c ) { delete[] c; }
  if ( idx1 ) { delete[] idx1; }
  if ( idx2 ) { delete[] idx2; }
  if ( idx3 ) { delete[] idx3; }
  if ( d_a ) { cudaFree(d_a); }
  if ( d_b ) { cudaFree(d_b); }
  if ( d_c ) { cudaFree(d_c); }
  if ( d_idx1 ) { cudaFree(d_idx1); }
  if ( d_idx2 ) { cudaFree(d_idx2); }
  if ( d_idx3 ) { cudaFree(d_idx3); }
  return true;
}

bool RS_CUDA::execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) {
  double startTime  = 0.0;
  double endTime    = 0.0;
  double runTime    = 0.0;
  double mbps       = 0.0;
  double flops      = 0.0;

  RSBaseImpl::RSKernelType kType = getKernelType();

  // TODO: run kernels
  switch(kType) {
    /* Sequential Kernels */
    case RSBaseImpl::RS_SEQ_COPY:
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;
      break;
    
    case RSBaseImpl::RS_SEQ_SCALE:
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SEQ_ADD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;
      break;

    case RSBaseImpl::RS_SEQ_TRIAD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;
      break;

    /* Gather kernels */
    case RSBaseImpl::RS_GATHER_COPY:
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;
      break;

    case RSBaseImpl::RS_GATHER_SCALE:
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;
      break;

    case RSBaseImpl::RS_GATHER_ADD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;
      break;

    case RSBaseImpl::RS_GATHER_TRIAD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;
      break;

    /* Scatter kernels */
    case RSBaseImpl::RS_SCATTER_COPY:
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_SCALE:
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_ADD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;
      break;

    case RSBaseImpl::RS_SCATTER_TRIAD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;
      break;

    /* Scatter-Gather kernels */
    case RSBaseImpl::RS_SG_COPY:
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3 streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;
      break;

    case RSBaseImpl::RS_SG_SCALE:
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, idx3 streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;
      break;

    case RSBaseImpl::RS_SG_ADD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, idx3 streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;
      break;

    case RSBaseImpl::RS_SG_TRIAD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, idx3, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;
      break;

    /* Central kernels */
    case RSBaseImpl::RS_CENTRAL_COPY:
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;
      break;
    
    case RSBaseImpl::RS_CENTRAL_SCALE:
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;
      break;

    case RSBaseImpl::RS_CENTRAL_ADD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;
      break;

    case RSBaseImpl::RS_CENTRAL_TRIAD:
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;
      break;

    /* All kernels */
    case RSBaseImpl::RS_ALL:
      /* RS_SEQ_COPY */
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
      TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;


      /* RS_SEQ_SCALE */
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;

      /* RS_SEQ_ADD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;

      /* RS_SEQ_TRIAD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      seqTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;

      /* RS_GATHER_COPY */
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
      TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;

      /* RS_GATHER_SCALE */
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;

      /* RS_GATHER_ADD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;

      /* RS_GATHER_TRIAD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      gatherTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;

      /* RS_SCATTER_COPY */
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;

      /* RS_SCATTER_SCALE */
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;

      /* RS_SCATTER_ADD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;

      /* RS_SCATTER_TRIAD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      scatterTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;

      /* RS_SG_COPY */
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3 streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
      TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
      MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_SG_COPY] = flops;

      /* RS_SG_SCALE */
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, idx3 streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
      TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;

      /* RS_SG_ADD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, d_idx1, d_idx2, idx3 streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
      TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
      MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_ADD] = flops;

      /* RS_SG_TRIAD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      sgTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, d_idx1, d_idx2, idx3, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;

      /* RS_CENTRAL_COPY */
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralCopy<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;

      /* RS_CENTRAL_SCALE */
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralScale<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;

      /* RS_CENTRAL_ADD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralAdd<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;

      /* RS_CENTRAL_TRIAD */
      cudaDeviceSynchronize();
      startTime = mySecond();
      centralTriad<<< threadBlocks, threadsPerBlock >>>(d_a, d_b, d_c, scalar, streamArraySize);
      cudaDeviceSynchronize();
      endTime = mySecond();
      runTime = calculateRunTime(startTime, endTime);
      mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
      TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
      MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
      FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;

      break;

    /* No kernel, something is wrong */
    default:
      std::cout << "RS_CUDA::execute() - ERROR: KERNEL NOT SET" << std::endl;
      return false;
  }

  return true;
}

#endif /* _ENABLE_CUDA_ */

/* EOF */
