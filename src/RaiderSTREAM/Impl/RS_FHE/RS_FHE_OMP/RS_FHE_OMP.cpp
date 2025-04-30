// _RS_FHE_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_FHE_OMP.h"

#ifdef _RS_FHE_OMP_H_

#include <chrono>
#include <iostream>
#include <omp.h>
#include <string>

#include "RSOpts.h"         // for streamArraySize, kernelName, etc.
#include "RS_FHE_Config.h"  // STREAM_TYPE, CreateCryptoContext, GenerateKeyPair
#include "ciphertext-ser.h" // for serialization headers
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

using namespace lbcrypto;

/**************************************************
 * @brief Constructor for the RS_FHE_OMP class.
 *
 * Initializes the RS_FHE_OMP object with the specified options.
 *
 * @param opts Options for the RS_FHE_OMP object.
 **************************************************/
RS_FHE_OMP::RS_FHE_OMP(const RSOpts &opts)
    : RSBaseImpl("RS_FHE_OMP",
                 opts.getKernelTypeFromName(opts.getKernelName())),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()), numPEs(opts.getNumPEs()),
      idx1(nullptr), idx2(nullptr), idx3(nullptr), scalar(3) {
  std::cout << "[RS_FHE_OMP] scheme = " << kernelName
            << ", arraySize = " << streamArraySize << ", threads = " << numPEs
            << std::endl;
}

/**************************************************
 * @brief Destructor for the RS_FHE_OMP class.
 **************************************************/
RS_FHE_OMP::~RS_FHE_OMP() {}

/**************************************************
 * @brief Allocates index arrays and initializes
 *        the FHE context, keys, and ciphertext buffers.
 *
 * @return True if allocation was successful.
 **************************************************/
bool RS_FHE_OMP::allocateData() {
  // 1) allocate and fill index arrays
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

  // 2) create FHE context & keys
#if defined(CKKS)
  cc = CreateCryptoContextCKKS(DEFAULT_DEPTH, DEFAULT_RING_DIM,
                               DEFAULT_SCALING_MOD_SIZE);
#elif defined(BFV)
  cc = CreateCryptoContextBFV(DEFAULT_DEPTH, DEFAULT_RING_DIM, DEFAULT_PTM);
#elif defined(BGV)
  cc = CreateCryptoContextBGV(DEFAULT_DEPTH, DEFAULT_RING_DIM, DEFAULT_PTM);
#else
#error "No encryption scheme defined!"
#endif
  kp = GenerateKeyPair(cc);

  // 3) allocate ciphertext buffers
  a_enc.resize(streamArraySize);
  b_enc.resize(streamArraySize);
  c_enc.resize(streamArraySize);

  // 4) chunked initialization (all-1s, all-2s, ...)
  size_t chunk = DEFAULT_CHUNK_SIZE;
  for (size_t off = 0; off < streamArraySize; off += chunk) {
    size_t n = std::min(chunk, streamArraySize - off);
    std::vector<STREAM_TYPE> A(n, STREAM_TYPE(1)), B(n, STREAM_TYPE(2)),
        C(n, STREAM_TYPE(1));
    Plaintext ptA = CreatePlaintextVector(cc, A);
    Plaintext ptB = CreatePlaintextVector(cc, B);
    Plaintext ptC = CreatePlaintextVector(cc, C);

    // encrypt each slot
    for (size_t i = 0; i < n; ++i) {
      a_enc[off + i] = cc->Encrypt(kp.publicKey, ptA);
      b_enc[off + i] = cc->Encrypt(kp.publicKey, ptB);
      c_enc[off + i] = cc->Encrypt(kp.publicKey, ptC);
    }
  }

  return true;
}

/**************************************************
 * @brief Frees allocated index arrays and clears
 *        ciphertext buffers.
 *
 * @return True always.
 **************************************************/
bool RS_FHE_OMP::freeData() {
  delete[] idx1;
  idx1 = nullptr;
  delete[] idx2;
  idx2 = nullptr;
  delete[] idx3;
  idx3 = nullptr;
  a_enc.clear();
  b_enc.clear();
  c_enc.clear();
  return true;
}

/**************************************************
 * @brief Executes the selected FHE‐enabled kernel.
 *
 * Mirrors RS_OMP::execute but invokes the FHE variants.
 *
 * @param TIMES     output array of runtimes
 * @param MBPS      output array of MB/s
 * @param FLOPS     output array of flop/s
 * @param BYTES     input bytes-per-iteration
 * @param FLOATOPS  input flops-per-iteration
 *
 * @return True if success.
 **************************************************/
bool RS_OMP_FHE::execute(double *TIMES, double *MBPS, double *FLOPS,
                         double *BYTES, double *FLOATOPS) {
  double startTime = 0.0, endTime = 0.0, runTime = 0.0;
  double mbps = 0.0, flops = 0.0;

  auto kType = getKernelType();

  switch (kType) {
  // ------------------------------
  // SEQUENTIAL KERNELS
  // ------------------------------
  case RSBaseImpl::RS_SEQ_COPY: {
    startTime = mySecond();
    seqCopyFHE(a_enc, b_enc, c_enc, streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SEQ_SCALE: {
    startTime = mySecond();
    seqScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SEQ_ADD: {
    startTime = mySecond();
    seqAddFHE(a_enc, b_enc, c_enc, streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SEQ_TRIAD: {
    startTime = mySecond();
    seqTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  // ------------------------------
  // GATHER KERNELS
  // ------------------------------
  case RSBaseImpl::RS_GATHER_COPY: {
    startTime = mySecond();
    gatherCopyFHE(a_enc, b_enc, c_enc,
                  std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                  streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_GATHER_SCALE: {
    startTime = mySecond();
    gatherScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc,
                   std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                   streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_GATHER_ADD: {
    startTime = mySecond();
    gatherAddFHE(
        a_enc, b_enc, c_enc, std::vector<ssize_t>(idx1, idx1 + streamArraySize),
        std::vector<ssize_t>(idx2, idx2 + streamArraySize), streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_GATHER_TRIAD: {
    startTime = mySecond();
    gatherTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc,
                   std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                   std::vector<ssize_t>(idx2, idx2 + streamArraySize),
                   streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  // ------------------------------
  // SCATTER KERNELS
  // ------------------------------
  case RSBaseImpl::RS_SCATTER_COPY: {
    startTime = mySecond();
    scatterCopyFHE(a_enc, b_enc, c_enc,
                   std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                   streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SCATTER_SCALE: {
    startTime = mySecond();
    scatterScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc,
                    std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                    streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SCATTER_ADD: {
    startTime = mySecond();
    scatterAddFHE(a_enc, b_enc, c_enc,
                  std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                  streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SCATTER_TRIAD: {
    startTime = mySecond();
    scatterTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc,
                    std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                    streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  // ------------------------------
  // SCATTER-GATHER KERNELS
  // ------------------------------
  case RSBaseImpl::RS_SG_COPY: {
    startTime = mySecond();
    sgCopyFHE(
        a_enc, b_enc, c_enc, std::vector<ssize_t>(idx1, idx1 + streamArraySize),
        std::vector<ssize_t>(idx2, idx2 + streamArraySize), streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SG_SCALE: {
    startTime = mySecond();
    sgScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc,
               std::vector<ssize_t>(idx1, idx1 + streamArraySize),
               std::vector<ssize_t>(idx2, idx2 + streamArraySize),
               streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SG_ADD: {
    startTime = mySecond();
    sgAddFHE(
        a_enc, b_enc, c_enc, std::vector<ssize_t>(idx1, idx1 + streamArraySize),
        std::vector<ssize_t>(idx2, idx2 + streamArraySize),
        std::vector<ssize_t>(idx3, idx3 + streamArraySize), streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SG_TRIAD: {
    startTime = mySecond();
    sgTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc,
               std::vector<ssize_t>(idx1, idx1 + streamArraySize),
               std::vector<ssize_t>(idx2, idx2 + streamArraySize),
               std::vector<ssize_t>(idx3, idx3 + streamArraySize),
               streamArraySize, scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  // ------------------------------
  // CENTRAL KERNELS
  // ------------------------------
  case RSBaseImpl::RS_CENTRAL_COPY: {
    startTime = mySecond();
    centralCopyFHE(a_enc, b_enc, c_enc, streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_CENTRAL_SCALE: {
    startTime = mySecond();
    centralScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, streamArraySize,
                    scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_CENTRAL_ADD: {
    startTime = mySecond();
    centralAddFHE(a_enc, b_enc, c_enc, streamArraySize);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_CENTRAL_TRIAD: {
    startTime = mySecond();
    centralTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, streamArraySize,
                    scalar);
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    mbps = calculateMBPS(BYTES[kType], runTime);
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  /* NO KERNELS, SOMETHING IS WRONG */
  default:
    std::cerr << "RS_OMP_FHE::execute() - ERROR: unknown kernel type\n";
    return false;
  }

  return true;
}

#endif /* _RS_FHE_OMP_H_ */
