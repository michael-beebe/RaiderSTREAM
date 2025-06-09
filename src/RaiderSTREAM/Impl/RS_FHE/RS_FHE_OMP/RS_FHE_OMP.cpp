// _RS_FHE_OMP_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RS_FHE_OMP.h"

// #ifdef _RS_FHE_OMP_H_

#include <chrono>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>

#include "RSOpts.h"         // for streamArraySize, kernelName, etc.
#include "RS_FHE_Config.h"  // STREAM_TYPE, CreateCryptoContext, GenerateKeyPair
#include "RS_FHE.h"
#include "ciphertext-ser.h" // for serialization headers
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

using namespace lbcrypto;
using RSFHE::CreatePlaintextVector;

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

    // Set OpenMP thread count from numPEs
    omp_set_num_threads(numPEs);
  std::string scheme;
#if defined(CKKS)
    scheme = "CKKS";
#elif defined(BFV)
    scheme = "BFV";
#elif defined(BGV)
    scheme = "BGV";
#else
    scheme = "UNKNOWN";
#endif
  std::cout << "[RS_FHE_OMP] scheme = " << scheme
            << ", arraySize = " << streamArraySize << ", threads = " << numPEs
            << ", OMP threads = " << omp_get_max_threads()
            << ", actual threads in parallel region = ";
  #pragma omp parallel
  {
    #pragma omp master
    std::cout << omp_get_num_threads() << std::endl;
  }
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
  std::cout << "[DEBUG] Entering allocateData()" << std::endl;
  // 1) allocate and fill index arrays
  idx1 = new ssize_t[streamArraySize];
    if (!idx1) {
    std::cerr << "[ERROR] Memory allocation for idx1 failed!" << std::endl;
    return false;
  }
  idx2 = new ssize_t[streamArraySize];
    if (!idx2) {
    std::cerr << "[ERROR] Memory allocation for idx2 failed!" << std::endl;
    return false;
  }
  idx3 = new ssize_t[streamArraySize];
  if (!idx3) {
    std::cerr << "[ERROR] Memory allocation for idx3 failed!" << std::endl;
    return false;
  }
  std::cout << "[DEBUG] Allocated index arrays" << std::endl;
#ifdef _ARRAYGEN_
  initReadIdxArray(idx1, streamArraySize, "RaiderSTREAM/arraygen/IDX1.txt");
  initReadIdxArray(idx2, streamArraySize, "RaiderSTREAM/arraygen/IDX2.txt");
  initReadIdxArray(idx3, streamArraySize, "RaiderSTREAM/arraygen/IDX3.txt");
  std::cout << "[DEBUG] Filled index arrays from files" << std::endl;
#else
  initRandomIdxArray(idx1, streamArraySize);
  initRandomIdxArray(idx2, streamArraySize);
  initRandomIdxArray(idx3, streamArraySize);
  std::cout << "[DEBUG] Filled index arrays with random values" << std::endl;
#endif

  // 2) create FHE context & keys
  std::cout << "[DEBUG] Creating FHE context..." << std::endl;
  cc = CreateCryptoContext();
  std::cout << "[DEBUG] FHE context created" << std::endl;
  kp = GenerateKeyPair(cc);
  std::cout << "[DEBUG] FHE key pair generated" << std::endl;

 // 3) allocate ciphertext buffers for number of chunks
size_t chunkSize = DEFAULT_CHUNK_SIZE;
size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
a_enc.resize(numChunks);
b_enc.resize(numChunks);
c_enc.resize(numChunks);
std::cout << "[DEBUG] Resized ciphertext buffers to numChunks = " << numChunks << std::endl;

std::cout << "[DEBUG] Starting chunked batch encryption with chunk size " << chunkSize << std::endl;
for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t chunk_start = chunk_idx * chunkSize;
    size_t chunk_end = std::min(chunk_start + chunkSize, static_cast<size_t>(streamArraySize));
    size_t currentChunkSize = chunk_end - chunk_start;

    // Sequential initialization (like other backends)
    std::vector<STREAM_TYPE> A(currentChunkSize), B(currentChunkSize), C(currentChunkSize);
    for (size_t i = 0; i < currentChunkSize; ++i) {
        size_t global_idx = chunk_start + i;
        // If using BGV/BFV, values will be reduced mod plaintext modulus automatically
        A[i] = static_cast<STREAM_TYPE>(global_idx % DEFAULT_PTM);
        B[i] = static_cast<STREAM_TYPE>(global_idx % DEFAULT_PTM);
        C[i] = static_cast<STREAM_TYPE>(global_idx % DEFAULT_PTM); 
    }

    // Create packed plaintexts for the chunk
    Plaintext ptA = CreatePlaintextVector(cc, A);
    Plaintext ptB = CreatePlaintextVector(cc, B);
    Plaintext ptC = CreatePlaintextVector(cc, C);

    // Encrypt the packed plaintexts
    a_enc[chunk_idx] = cc->Encrypt(kp.publicKey, ptA);
    b_enc[chunk_idx] = cc->Encrypt(kp.publicKey, ptB);
    c_enc[chunk_idx] = cc->Encrypt(kp.publicKey, ptC);

    std::cout << "[DEBUG] Encrypted chunk " << (chunk_idx + 1)
              << " (" << currentChunkSize << " elements)" << std::endl;

    // DEBUG: decrypt and print first 100 elements of A vs. decrypted A
    if (chunk_idx == 0) {
      Plaintext ptA_dec;
      cc->Decrypt(kp.secretKey, a_enc[chunk_idx], &ptA_dec);
      ptA_dec->SetLength(ptA->GetLength());
      #if defined(CKKS)
      auto decA = ptA_dec->GetCKKSPackedValue();
      #else
      auto decA = ptA_dec->GetPackedValue();
      #endif

      std::cout << "[DEBUG] Chunk 0  Plain A[0..99]: ";
      for (size_t i = 0; i < std::min<size_t>(100, A.size()); ++i)
        std::cout << A[i] << ' ';
      std::cout << "\n[DEBUG] Chunk 0 Decr A[0..99]: ";
      for (size_t i = 0; i < std::min<size_t>(100, decA.size()); ++i)
        std::cout << decA[i] << ' ';
      std::cout << std::endl;
    }
}
  std::cout << "[DEBUG] Finished allocateData()" << std::endl;
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
bool RS_FHE_OMP::execute(double *TIMES, double *MBPS, double *FLOPS,
                         double *BYTES, double *FLOATOPS) {
  double startTime = 0.0, endTime = 0.0, runTime = 0.0;
  double mbps = 0.0, flops = 0.0;
  size_t chunkSize = DEFAULT_CHUNK_SIZE;

  auto kType = getKernelType();
  std::cout << "[DEBUG] Entering execute(). Kernel type: " << kType << " (" << kernelName << ")" << std::endl;

  // Handle RS_ALL case by running all kernels
  if (kType == RSBaseImpl::RS_ALL) {
    for (int k = static_cast<int>(RSBaseImpl::RS_SEQ_COPY); k < static_cast<int>(RSBaseImpl::RS_ALL); ++k) {
      RSBaseImpl::RSKernelType currentKernel = static_cast<RSBaseImpl::RSKernelType>(k);
      std::cout << "[DEBUG] Running kernel: " << BenchTypeTable[k].Notes << std::endl;
      
      // Run the specific kernel
      if (!executeKernel(currentKernel, TIMES, MBPS, FLOPS, BYTES, FLOATOPS, chunkSize)) {
        std::cerr << "RS_FHE_OMP::execute() - ERROR: failed to execute kernel " << k << std::endl;
        return false;
      }
    }
    return true;
  }

  // Single kernel execution
  return executeKernel(kType, TIMES, MBPS, FLOPS, BYTES, FLOATOPS, chunkSize);
}

bool RS_FHE_OMP::executeKernel(RSBaseImpl::RSKernelType kType, double *TIMES,
                               double *MBPS, double *FLOPS, double *BYTES,
                               double *FLOATOPS, size_t chunkSize) {
  double startTime = 0.0, endTime = 0.0, runTime = 0.0;
  double mbps = 0.0, flops = 0.0;

  switch (kType) {
  // ------------------------------
  // SEQUENTIAL KERNELS
  // ------------------------------
  case RSBaseImpl::RS_SEQ_COPY: {
    startTime = mySecond();
    std::cout << "[DEBUG] Calling seqCopyFHE(a_enc, b_enc, c_enc, " << chunkSize << ", " << streamArraySize << ")" << std::endl;
    seqCopyFHE(a_enc, b_enc, c_enc, chunkSize, streamArraySize);
    std::cout << "[DEBUG] Finished seqCopyFHE" << std::endl;
    endTime = mySecond();
    runTime = calculateRunTime(startTime, endTime);
    std::cout << "[DEBUG] calculateMBPS(" << BYTES[kType] << ", " << runTime << ") = " << calculateMBPS(BYTES[kType], runTime) << std::endl;
    std::cout << "[DEBUG] calculateFLOPS(" << FLOATOPS[kType] << ", " << runTime << ") = " << calculateFLOPS(FLOATOPS[kType], runTime) << std::endl;
    mbps = calculateMBPS(BYTES[kType], runTime);
    std::cout << "[DEBUG] MBPS: " << mbps << std::endl;
    flops = calculateFLOPS(FLOATOPS[kType], runTime);
    std::cout << "[DEBUG] FLOPS: " << flops << std::endl;
    TIMES[kType] = runTime;
    MBPS[kType] = mbps;
    FLOPS[kType] = flops;
    break;
  }

  case RSBaseImpl::RS_SEQ_SCALE: {
    startTime = mySecond();
    seqScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, chunkSize, streamArraySize, scalar);
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
    seqAddFHE(cc, a_enc, b_enc, c_enc, chunkSize, streamArraySize);
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
    seqTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, chunkSize, streamArraySize, scalar);
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
                  chunkSize, streamArraySize);
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
                   chunkSize, streamArraySize, scalar);
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
        cc, a_enc, b_enc, c_enc, std::vector<ssize_t>(idx1, idx1 + streamArraySize),
        std::vector<ssize_t>(idx2, idx2 + streamArraySize), chunkSize, streamArraySize);
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
                   chunkSize, streamArraySize, scalar);
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
                   chunkSize, streamArraySize);
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
                    chunkSize, streamArraySize, scalar);
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
    scatterAddFHE(cc, a_enc, b_enc, c_enc,
                  std::vector<ssize_t>(idx1, idx1 + streamArraySize),
                  chunkSize, streamArraySize);
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
                    chunkSize, streamArraySize, scalar);
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
        std::vector<ssize_t>(idx2, idx2 + streamArraySize), chunkSize, streamArraySize);
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
               chunkSize, streamArraySize, scalar);
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
        cc, a_enc, b_enc, c_enc, std::vector<ssize_t>(idx1, idx1 + streamArraySize),
        std::vector<ssize_t>(idx2, idx2 + streamArraySize),
        std::vector<ssize_t>(idx3, idx3 + streamArraySize), chunkSize, streamArraySize);
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
               chunkSize, streamArraySize, scalar);
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
    centralCopyFHE(a_enc, b_enc, c_enc, chunkSize, streamArraySize);
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
    centralScaleFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, chunkSize, streamArraySize, scalar);
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
    centralAddFHE(cc, a_enc, b_enc, c_enc, chunkSize, streamArraySize);
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
    centralTriadFHE(cc, kp.publicKey, a_enc, b_enc, c_enc, chunkSize, streamArraySize, scalar);
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
    std::cerr << "RS_FHE_OMP::execute() - ERROR: unknown kernel type\n";
    return false;
  }

  return true;
}

// #endif /* _RS_FHE_OMP_H_ */
