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

  // FIXME: maybe let's do this in RS_Main.cpp
  // MPI_Init( NULL, NULL );
  // MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  // MPI_Comm_size(MPI_COMM_WORLD,&size);
  // MPI_Barrier(MPI_COMM_WORLD);

  /* Calculate the chunk size for each rank */
  ssize_t chunkSize  = streamArraySize / size;
  ssize_t remainder   = streamArraySize % size;


  /* Adjust the chunk size for the last process */
  if ( myRank == size - 1 ) {
    chunkSize += remainder;
  }

  /* Allocate memory for the local chunks */
  MPI_Alloc_mem(chunkSize * sizeof(double), MPI_INFO_NULL, &a);
  MPI_Alloc_mem(chunkSize * sizeof(double), MPI_INFO_NULL, &b);
  MPI_Alloc_mem(chunkSize * sizeof(double), MPI_INFO_NULL, &c);
  MPI_Alloc_mem(chunkSize * sizeof(ssize_t), MPI_INFO_NULL, &idx1);
  MPI_Alloc_mem(chunkSize * sizeof(ssize_t), MPI_INFO_NULL, &idx2);
  MPI_Alloc_mem(chunkSize * sizeof(ssize_t), MPI_INFO_NULL, &idx3);

  // a     = new  double[chunkSize];
  // b     = new  double[chunkSize];
  // c     = new  double[chunkSize];
  // idx1  = new ssize_t[chunkSize];
  // idx2  = new ssize_t[chunkSize];
  // idx3  = new ssize_t[chunkSize];

  /* Initialize the local chunks */
  initStreamArray(a, chunkSize, 1.0);
  initStreamArray(b, chunkSize, 2.0);
  initStreamArray(c, chunkSize, 0.0);
  initRandomIdxArray(idx1, chunkSize);
  initRandomIdxArray(idx2, chunkSize);
  initRandomIdxArray(idx3, chunkSize);

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool RS_MPI_OMP::freeData() {
  if ( a ) { MPI_Free_mem( a ); }
  if ( b ) { MPI_Free_mem( b ); }
  if ( c ) { MPI_Free_mem( c ); }
  if ( idx1 ) { MPI_Free_mem( idx1 ); }
  if ( idx2 ) { MPI_Free_mem( idx2 ); }
  if ( idx3 ) { MPI_Free_mem( idx3 ); }
  return true;
}

bool RS_MPI_OMP::execute(
  double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS
) {
  // TODO: RS_MPI_OMP::execute

  // TODO: disperse chunks across ranks

  // TODO: Execute each kernel

  return true;
}


#endif /* _RS_MPI_OMP_H_ */

/* EOF */

