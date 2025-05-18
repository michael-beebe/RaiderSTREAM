// RS_FHE_OMP.h
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details

#ifndef RS_FHE_OMP_H
#define RS_FHE_OMP_H

#include <vector>
#include <cstddef>
#include <omp.h>

#include "RSOpts.h"           // for RSOpts
#include "RSBaseImpl.h"       // for RSBaseImpl
#include "RS_FHE_Config.h"    // for STREAM_TYPE, CreateCryptoContext, GenerateKeyPair
#include "openfhe.h"          // for CryptoContext, Ciphertext, DCRTPoly, PublicKey

using namespace lbcrypto;

/**
 * @brief RaiderSTREAM OpenMP + FHE implementation class
 *
 * This class provides a fully-homomorphic-encrypted version of the
 * RaiderSTREAM benchmark, using OpenMP for parallelism and OpenFHE
 * for encryption/operations.  It lives entirely under RS_FHE/ so it
 * does not touch the other backends.
 */
class RS_OMP_FHE : public RSBaseImpl {
public:
  /**
   * @brief Constructor
   * @param opts  Command-line options (kernel name, array size, etc.)
   */
  RS_OMP_FHE(const RSOpts &opts);

  /** @brief Destructor */
  ~RS_OMP_FHE();

  /**
   * @brief Allocate and initialize all data structures
   *        (index arrays, CryptoContext, keys, ciphertext arrays).
   * @return true on success
   */
  bool allocateData() override;

  /**
   * @brief Run the selected kernel (all 20 variants) in encrypted form.
   * @param TIMES     Output array for execution times per kernel
   * @param MBPS      Output array for MB/s per kernel
   * @param FLOPS     Output array for FLOPS per kernel
   * @param BYTES     Input array of byte-counts per kernel
   * @param FLOATOPS  Input array of flop-counts per kernel
   * @return true on success
   */
  bool execute(double *TIMES,
               double *MBPS,
               double *FLOPS,
               double *BYTES,
               double *FLOATOPS) override;

  /**
   * @brief Free all allocated resources.
   * @return true on success
   */
  bool freeData() override;

private:
  std::string                     kernelName;      ///< name of kernel to run
  ssize_t                         streamArraySize; ///< total array length
  int                             numPEs;          ///< number of OpenMP threads
  ssize_t                        *idx1, *idx2, *idx3; ///< plaintext index arrays
  STREAM_TYPE                    scalar;          ///< scale factor for scale/triad
  CryptoContext<DCRTPoly> cc;  ///< FHE crypto context
  KeyPair<DCRTPoly>       kp;  ///< FHE key pair
  std::vector<Ciphertext<DCRTPoly>>
                                  a_enc, b_enc, c_enc; ///< encrypted data arrays
};

// -----------------------------------------------------------------------------
// Declarations of the 20 FHE‐enabled RaiderSTREAM kernels.
// Each one operates on vectors of Ciphertext<DCRTPoly> with OpenMP.
// -----------------------------------------------------------------------------

/// @brief Homomorphic copy: c_enc[j] = a_enc[j]
void seqCopyFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize);

/// @brief Homomorphic scale: b_enc[j] = scalar * c_enc[j]
void seqScaleFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

/// @brief Homomorphic add: c_enc[j] = a_enc[j] + b_enc[j]
void seqAddFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize);

/// @brief Homomorphic triad: a_enc[j] = b_enc[j] + scalar * c_enc[j]
void seqTriadFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

// Gather kernels
void gatherCopyFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx,
  ssize_t streamArraySize);

void gatherScaleFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

void gatherAddFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx1,
  const std::vector<ssize_t> &idx2,
  ssize_t streamArraySize);

void gatherTriadFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx1,
  const std::vector<ssize_t> &idx2,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

// Scatter kernels
void scatterCopyFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx,
  ssize_t streamArraySize);

void scatterScaleFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

void scatterAddFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx,
  ssize_t streamArraySize);

void scatterTriadFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

// Scatter-gather kernels
void sgCopyFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx1,
  const std::vector<ssize_t> &idx2,
  ssize_t streamArraySize);

void sgScaleFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx1,
  const std::vector<ssize_t> &idx2,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

void sgAddFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx1,
  const std::vector<ssize_t> &idx2,
  const std::vector<ssize_t> &idx3,
  ssize_t streamArraySize);

void sgTriadFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  const std::vector<ssize_t> &idx1,
  const std::vector<ssize_t> &idx2,
  const std::vector<ssize_t> &idx3,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

// Central kernels
void centralCopyFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize);

void centralScaleFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

void centralAddFHE(
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize);

void centralTriadFHE(
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &a_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &b_enc,
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &c_enc,
  ssize_t streamArraySize,
  STREAM_TYPE scalar);

#endif // RS_FHE_OMP_H
