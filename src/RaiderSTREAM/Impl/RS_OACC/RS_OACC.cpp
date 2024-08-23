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

#include <stdio.h>

RS_OACC::RS_OACC(const RSOpts &opts)
    : RSBaseImpl("RS_OACC", opts.getKernelTypeFromName(opts.getKernelName())),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()), numPEs(opts.getNumPEs()),
      numGangs(opts.getThreadBlocks()), numWorkers(opts.getThreadsPerBlocks()),
      lArgc(0), lArgv(nullptr), a(nullptr), b(nullptr), c(nullptr),
      d_a(nullptr), d_b(nullptr), d_c(nullptr), idx1(nullptr), idx2(nullptr),
      idx3(nullptr), d_idx1(nullptr), d_idx2(nullptr), d_idx3(nullptr),
      scalar(3.0) {}

RS_OACC::~RS_OACC() {}

bool RS_OACC::setDevice() {
  // std::cout <<"OpenACC version: "<< _OPENACC <<std::endl;
  acc_device_t device = acc_get_device_type();
  std::cout << "The device type is: " << device << std::endl;
  acc_set_device_type(device); // Device type is nvidia by default, must be
                               // changed if not running on a nvidia device
  acc_init(device);
  std::cout << "The name of the device we are using is: "
            << acc_get_property_string(0, device, acc_property_name)
            << std::endl;
  int gangs = 0;
  int workers = 0;
#pragma acc parallel num_gangs(numGangs) reduction(+ : gangs)
  { gangs++; }
  std::cout << "Number of Gangs: " << gangs << std::endl;
  return true;
}

bool RS_OACC::allocateData() {

  ssize_t streamMemArraySize = streamArraySize * sizeof(double);
  ssize_t idxMemArraySize = streamArraySize * sizeof(ssize_t);

  a = new double[streamArraySize];
  b = new double[streamArraySize];
  c = new double[streamArraySize];
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

  /* a -> d_a */
  d_a = (double *)acc_malloc(streamMemArraySize);
  if (d_a == nullptr) {
    std::cerr << "RS_OACC::allocateData: 'd_a' could not be allocated on device"
              << std::endl;
    free(a);
    free(b);
    free(c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  acc_memcpy_to_device(d_a, a, streamMemArraySize);

  /* b -> d_b */
  d_b = (double *)acc_malloc(streamMemArraySize);
  if (d_b == nullptr) {
    std::cerr << "RS_OACC:allocateData: 'd_b' could not be allocated on device";
    free(a);
    free(b);
    free(c);
    acc_free(d_a);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  acc_memcpy_to_device(d_b, b, streamMemArraySize);

  /* c -> d_c */
  d_c = (double *)acc_malloc(streamMemArraySize);
  if (d_c == nullptr) {
    std::cerr << "RS_OACC:allocateData: 'd_c' could not be allocated on device";
    free(a);
    free(b);
    free(c);
    acc_free(d_a);
    acc_free(d_b);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  acc_memcpy_to_device(d_c, c, streamMemArraySize);

  /* idx1 -> d_idx1 */
  d_idx1 = (ssize_t *)acc_malloc(idxMemArraySize);
  if (d_idx1 == nullptr) {
    std::cerr
        << "RS_OACC:allocateData: 'd_idx1' could not be allocated on device";
    free(a);
    free(b);
    free(c);
    acc_free(d_a);
    acc_free(d_b);
    acc_free(d_c);
    free(idx1);
    free(idx2);
    free(idx3);
    return false;
  }
  acc_memcpy_to_device(d_idx1, idx1, idxMemArraySize);

  /* idx2 -> d_idx2 */
  d_idx2 = (ssize_t *)acc_malloc(idxMemArraySize);
  if (d_idx2 == nullptr) {
    std::cerr
        << "RS_OACC:allocateData: 'd_idx2' could not be allocated on device";
    free(a);
    free(b);
    free(c);
    acc_free(d_a);
    acc_free(d_b);
    acc_free(d_c);
    free(idx1);
    free(idx2);
    free(idx3);
    acc_free(d_idx1);
    return false;
  }
  acc_memcpy_to_device(d_idx2, idx2, idxMemArraySize);

  /* idx3 -> d_idx3 */
  d_idx3 = (ssize_t *)acc_malloc(idxMemArraySize);
  if (d_idx3 == nullptr) {
    std::cerr
        << "RS_OACC:allocateData: 'd_idx3' could not be allocated on device";
    free(a);
    free(b);
    free(c);
    acc_free(d_a);
    acc_free(d_b);
    acc_free(d_c);
    free(idx1);
    free(idx2);
    free(idx3);
    acc_free(d_idx1);
    acc_free(d_idx2);
    return false;
  }
  acc_memcpy_to_device(d_idx3, idx3, idxMemArraySize);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /* Verify that copying data back and forth works as expected */
  double *test_a = new double[streamArraySize];
  double *test_b = new double[streamArraySize];
  double *test_c = new double[streamArraySize];
  ssize_t *test_idx1 = new ssize_t[streamArraySize];
  ssize_t *test_idx2 = new ssize_t[streamArraySize];
  ssize_t *test_idx3 = new ssize_t[streamArraySize];

  acc_memcpy_from_device(test_a, d_a, streamMemArraySize);
  acc_memcpy_from_device(test_b, d_b, streamMemArraySize);
  acc_memcpy_from_device(test_c, d_c, streamMemArraySize);
  acc_memcpy_from_device(test_idx1, d_idx1, idxMemArraySize);
  acc_memcpy_from_device(test_idx2, d_idx2, idxMemArraySize);
  acc_memcpy_from_device(test_idx3, d_idx3, idxMemArraySize);

  bool success = true;
  for (ssize_t i = 0; i < streamArraySize; ++i) {
    if (a[i] != test_a[i]) {
      std::cerr << "RS_OACC::allocateData: Data verification failed at index "
                << i << " for array 'a'" << std::endl;
      success = false;
      break;
    }
    if (b[i] != test_b[i]) {
      std::cerr << "RS_OACC::allocateData: Data verification failed at index "
                << i << " for array 'b'" << std::endl;
      success = false;
      break;
    }
    if (c[i] != test_c[i]) {
      std::cerr << "RS_OACC::allocateData: Data verification failed at index "
                << i << " for array 'c'" << std::endl;
      success = false;
      break;
    }
    if (idx1[i] != test_idx1[i]) {
      std::cerr << "RS_OACC::allocateData: Data verification failed at index "
                << i << " for array 'idx1'" << std::endl;
      success = false;
      break;
    }
    if (idx2[i] != test_idx2[i]) {
      std::cerr << "RS_OACC::allocateData: Data verification failed at index "
                << i << " for array 'idx2'" << std::endl;
      success = false;
      break;
    }
    if (idx2[i] != test_idx2[i]) {
      std::cerr << "RS_OACC::allocateData: Data verification failed at index "
                << i << " for array 'idx3'" << std::endl;
      success = false;
      break;
    }
  }
  delete[] test_a;
  delete[] test_b;
  delete[] test_c;
  if (!success) {
    free(a);
    free(b);
    free(c);
    acc_free(d_a);
    acc_free(d_b);
    acc_free(d_c);
    free(idx1);
    free(idx2);
    free(idx3);
    acc_free(d_idx1);
    acc_free(d_idx2);
    acc_free(d_idx3);
    return false;
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _DEBUG_
  std::cout << "==============================================================="
               "===================="
            << std::endl;
  std::cout << " RaiderSTREAM Array Info:" << std::endl;
  std::cout << "==============================================================="
               "===================="
            << std::endl;
  std::cout << "streamArraySize         = " << streamArraySize << std::endl;
  std::cout << "a[streamArraySize-1]    = " << a[streamArraySize - 1]
            << std::endl;
  std::cout << "b[streamArraySize-1]    = " << b[streamArraySize - 1]
            << std::endl;
  std::cout << "c[streamArraySize-1]    = " << c[streamArraySize - 1]
            << std::endl;
  std::cout << "idx1[streamArraySize-1] = " << idx1[streamArraySize - 1]
            << std::endl;
  std::cout << "idx2[streamArraySize-1] = " << idx2[streamArraySize - 1]
            << std::endl;
  std::cout << "idx3[streamArraySize-1] = " << idx3[streamArraySize - 1]
            << std::endl;
  std::cout << "==============================================================="
               "===================="
            << std::endl;
#endif

  return true;
}

bool RS_OACC::freeData() {
  if (a) {
    delete[] a;
  }
  if (b) {
    delete[] b;
  }
  if (c) {
    delete[] c;
  }
  if (idx1) {
    delete[] idx1;
  }
  if (idx2) {
    delete[] idx2;
  }
  if (idx3) {
    delete[] idx3;
  }
  if (d_a) {
    acc_free(d_a);
  }
  if (d_b) {
    acc_free(d_b);
  }
  if (d_c) {
    acc_free(d_c);
  }
  if (d_idx1) {
    acc_free(d_idx1);
  }
  if (d_idx2) {
    acc_free(d_idx2);
  }
  if (d_idx3) {
    acc_free(d_idx3);
  }
  return true;
}

bool RS_OACC::execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES,
                      double *FLOATOPS) {
  double startTime = 0.0;
  double endTime = 0.0;
  double runTime = 0.0;
  double mbps = 0.0;
  double flops = 0.0;

  RSBaseImpl::RSKernelType kType = getKernelType();

  switch (kType) {
  /* SEQUENTIAL KERNELS */
  case RSBaseImpl::RS_SEQ_COPY:
    runTime = seqCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
    TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;
    break;

  case RSBaseImpl::RS_SEQ_SCALE:
    runTime =
        seqScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;
    break;

  case RSBaseImpl::RS_SEQ_ADD:
    runTime = seqAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;
    break;

  case RSBaseImpl::RS_SEQ_TRIAD:
    runTime =
        seqTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;
    break;

  /* GATHER KERNELS */
  case RSBaseImpl::RS_GATHER_COPY:
    runTime = gatherCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                         streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
    TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;
    break;

  case RSBaseImpl::RS_GATHER_SCALE:
    runTime = gatherScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                          streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;
    break;

  case RSBaseImpl::RS_GATHER_ADD:
    runTime = gatherAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                        streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;
    break;

  case RSBaseImpl::RS_GATHER_TRIAD:
    runTime = gatherTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                          streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;
    break;

  /* SCATTER KERNELS */
  case RSBaseImpl::RS_SCATTER_COPY:
    runTime = scatterCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                          streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;
    break;

  case RSBaseImpl::RS_SCATTER_SCALE:
    runTime = scatterScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                           streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;
    break;

  case RSBaseImpl::RS_SCATTER_ADD:
    runTime = scatterAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                         streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;
    break;

  case RSBaseImpl::RS_SCATTER_TRIAD:
    runTime = scatterTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                           streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;
    break;

  /* SCATTER-GATHER KERNELS */
  case RSBaseImpl::RS_SG_COPY:
    runTime = sgCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                     streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
    TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SG_COPY] = flops;
    break;

  case RSBaseImpl::RS_SG_SCALE:
    runTime = sgScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                      streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;
    break;

  case RSBaseImpl::RS_SG_ADD:
    runTime = sgAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, d_idx3,
                    streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
    TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_ADD] = flops;
    break;

  case RSBaseImpl::RS_SG_TRIAD:
    runTime = sgTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                      d_idx3, streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;
    break;

  /* CENTRAL KERNELS */
  case RSBaseImpl::RS_CENTRAL_COPY:
    runTime = centralCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;
    break;

  case RSBaseImpl::RS_CENTRAL_SCALE:
    runTime = centralScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize,
                           scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;
    break;

  case RSBaseImpl::RS_CENTRAL_ADD:
    runTime = centralAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;
    break;

  case RSBaseImpl::RS_CENTRAL_TRIAD:
    runTime = centralTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize,
                           scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = flops;
    break;

  /* ALL KERNELS */
  case RSBaseImpl::RS_ALL:
    /* RS_SEQ_COPY */
    runTime = seqCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_COPY], runTime);
    TIMES[RSBaseImpl::RS_SEQ_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_COPY] = flops;

    /* RS_SEQ_SCALE */
    runTime =
        seqScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SEQ_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_SCALE] = flops;

    /* RS_SEQ_ADD */
    runTime = seqAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_ADD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_ADD] = flops;

    /* RS_SEQ_TRIAD */
    runTime =
        seqTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SEQ_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SEQ_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SEQ_TRIAD] = flops;

    /* RS_GATHER_COPY */
    runTime = gatherCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                         streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_COPY], runTime);
    TIMES[RSBaseImpl::RS_GATHER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_COPY] = flops;

    /* RS_GATHER_SCALE */
    runTime = gatherScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                          streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_GATHER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_SCALE] = flops;

    /* RS_GATHER_ADD */
    runTime = gatherAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                        streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_ADD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_ADD] = flops;

    /* RS_GATHER_TRIAD */
    runTime = gatherTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                          streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_GATHER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_GATHER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_GATHER_TRIAD] = flops;

    /* RS_SCATTER_COPY */
    runTime = scatterCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                          streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_COPY], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_COPY] = flops;

    /* RS_SCATTER_SCALE */
    runTime = scatterScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                           streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_SCALE] = flops;

    /* RS_SCATTER_ADD */
    runTime = scatterAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                         streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_ADD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_ADD] = flops;

    /* RS_SCATTER_TRIAD */
    runTime = scatterTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1,
                           streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SCATTER_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SCATTER_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SCATTER_TRIAD] = flops;

    /* RS_SG_COPY */
    runTime = sgCopy(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                     streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_COPY], runTime);
    TIMES[RSBaseImpl::RS_SG_COPY] = runTime;
    MBPS[RSBaseImpl::RS_SG_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_SG_COPY] = flops;

    /* RS_SG_SCALE */
    runTime = sgScale(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                      streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_SCALE], runTime);
    TIMES[RSBaseImpl::RS_SG_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_SG_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_SG_SCALE] = flops;

    /* RS_SG_ADD */
    runTime = sgAdd(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2, d_idx3,
                    streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_ADD], runTime);
    TIMES[RSBaseImpl::RS_SG_ADD] = runTime;
    MBPS[RSBaseImpl::RS_SG_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_ADD] = flops;

    /* RS_SG_TRIAD */
    runTime = sgTriad(numGangs, numWorkers, d_a, d_b, d_c, d_idx1, d_idx2,
                      d_idx3, streamArraySize, scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_SG_TRIAD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_SG_TRIAD], runTime);
    TIMES[RSBaseImpl::RS_SG_TRIAD] = runTime;
    MBPS[RSBaseImpl::RS_SG_TRIAD] = mbps;
    FLOPS[RSBaseImpl::RS_SG_TRIAD] = flops;

    /* RS_CENTRAL_COPY */
    runTime = centralCopy(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_COPY] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_COPY] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_COPY] = flops;

    /* RS_CENTRAL_SCALE */
    runTime = centralScale(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize,
                           scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_SCALE] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_SCALE] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_SCALE] = flops;

    /* RS_CENTRAL_ADD */
    runTime = centralAdd(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    flops = calculateFLOPS(FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD], runTime);
    TIMES[RSBaseImpl::RS_CENTRAL_ADD] = runTime;
    MBPS[RSBaseImpl::RS_CENTRAL_ADD] = mbps;
    FLOPS[RSBaseImpl::RS_CENTRAL_ADD] = flops;

    /* RS_CENTRAL_TRIAD */
    runTime = centralTriad(numGangs, numWorkers, d_a, d_b, d_c, streamArraySize,
                           scalar);
    endTime = mySecond();
    // runTime = calculateRunTime(startTime, endTime);
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
