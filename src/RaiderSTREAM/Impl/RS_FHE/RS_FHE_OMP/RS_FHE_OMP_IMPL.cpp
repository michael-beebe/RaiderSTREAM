// RS_FHE_OMP_IMPL.cpp
// -----------------------------------------------------------------------------
// Fully‑Homomorphic‑Encryption OpenMP kernels for RaiderSTREAM
// -----------------------------------------------------------------------------

#include "RS_FHE_OMP.h"
#include "RS_FHE_Config.h"
#include "RS_FHE.h"  

#include <omp.h>
#include <vector>

using lbcrypto::CryptoContext;
using lbcrypto::PublicKey;
using lbcrypto::Ciphertext;
using lbcrypto::DCRTPoly;
using RSFHE::CreatePlaintextValue;
using RSFHE::EvalAddOperation;

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “copy” kernel (sequential STREAM copy).
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b (unused)
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
 * @param streamArraySize Number of elements in the arrays
 */
void seqCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize)
{
    (void)b_enc;  // unused
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = a_enc[chunk_idx];
    }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “scale” kernel (sequential STREAM scale).
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc Encrypted input array a (unused)
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param chunkSize Size of each chunk
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar to multiply
 */
void seqScaleFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
    (void)a_enc;  // unused
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    auto scalar_pt = CreatePlaintextValue(cc, scalar);
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        b_enc[chunk_idx] = cc->EvalMult(c_enc[chunk_idx], scalar_pt);
    }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “add” kernel (sequential STREAM add).
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
 * @param streamArraySize Number of elements
 */
void seqAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize)
{
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = EvalAddOperation(cc, a_enc[chunk_idx], b_enc[chunk_idx]);
    }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “triad” kernel (sequential STREAM triad).
 *        Computes a = b + scalar * c.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc    Encrypted output array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted input array c
 * @param chunkSize Size of each chunk
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar to multiply
 */
void seqTriadFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    auto scalar_pt = CreatePlaintextValue(cc, scalar);
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        auto tmp = cc->EvalMult(c_enc[chunk_idx], scalar_pt);
        a_enc[chunk_idx] = EvalAddOperation(cc, b_enc[chunk_idx], tmp);
    }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑copy kernel.
 * @param a_enc    Encrypted input array a
 * @param b_enc Unused
 * @param c_enc    Encrypted output array c
 * @param idx1     Index array for gather
 * @param streamArraySize Number of elements
 */
void gatherCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[j] = a_enc[idx1[j]];
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑scale kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc Unused
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param idx1     Index array for gather
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void gatherScaleFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    b_enc[j] = cc->EvalMult(c_enc[idx1[j]], scalar_pt);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param idx1     First index array
 * @param idx2     Second index array
 * @param streamArraySize Number of elements
 */
void gatherAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  ssize_t streamArraySize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[j] = EvalAddOperation(cc, a_enc[idx1[j]], b_enc[idx2[j]]);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑triad kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc    Encrypted output array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted input array c
 * @param idx1     First index array
 * @param idx2     Second index array
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void gatherTriadFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    auto tmp      = cc->EvalMult(c_enc[idx2[j]], scalar_pt);
    a_enc[j]      = EvalAddOperation(cc, b_enc[idx1[j]], tmp);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑copy kernel.
 * @param a_enc    Encrypted input array a
 * @param b_enc Unused
 * @param c_enc    Encrypted output array c
 * @param idx1     Index array for scatter
 * @param streamArraySize Number of elements
 */
void scatterCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[idx1[j]] = a_enc[j];
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑scale kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc Unused
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param idx1     Index array for scatter
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void scatterScaleFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    b_enc[idx1[j]] = cc->EvalMult(c_enc[j], scalar_pt);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param idx1     Index array for scatter
 * @param streamArraySize Number of elements
 */
void scatterAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  ssize_t streamArraySize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[idx1[j]] = EvalAddOperation(cc, a_enc[j], b_enc[j]);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑triad kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc    Encrypted output array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted input array c
 * @param idx1     Index array for scatter
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void scatterTriadFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    auto tmp = cc->EvalMult(c_enc[j], scalar_pt);
    a_enc[idx1[j]] = EvalAddOperation(cc, b_enc[j], tmp);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑copy kernel.
 * @param a_enc    Encrypted input array a
 * @param b_enc Unused
 * @param c_enc    Encrypted output array c
 * @param idx1     First index array
 * @param idx2     Second index array
 * @param streamArraySize Number of elements
 */
void sgCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[idx1[j]] = a_enc[idx2[j]];
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑scale kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc Unused
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param idx1     First index array
 * @param idx2     Second index array
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void sgScaleFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    b_enc[idx2[j]] = cc->EvalMult(c_enc[idx1[j]], scalar_pt);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param idx1     First index array
 * @param idx2     Second index array
 * @param idx3     Third index array
 * @param streamArraySize Number of elements
 */
void sgAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  const std::vector<ssize_t>& idx3,
  ssize_t streamArraySize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[idx1[j]] = EvalAddOperation(cc, a_enc[idx2[j]], b_enc[idx3[j]]);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑triad kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc    Encrypted output array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted input array c
 * @param idx1     First index array
 * @param idx2     Second index array
 * @param idx3     Third index array
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void sgTriadFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  const std::vector<ssize_t>& idx3,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    auto tmp      = cc->EvalMult(c_enc[idx1[j]], scalar_pt);
    a_enc[idx2[j]] = EvalAddOperation(cc, b_enc[idx3[j]], tmp);
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “central” copy kernel.
 *        All outputs get the encryption of a[0].
 * @param a_enc    Encrypted input array a
 * @param b_enc Unused
 * @param c_enc    Encrypted output array c
 * @param streamArraySize Number of elements
 */
void centralCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[j] = a_enc[0];
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “central” scale kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc Unused
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void centralScaleFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  auto base      = cc->EvalMult(c_enc[0], scalar_pt);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    b_enc[j] = base;
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “central” add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param streamArraySize Number of elements
 */
void centralAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  ssize_t streamArraySize)
{
  auto sum = EvalAddOperation(cc, a_enc[0], b_enc[0]);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    c_enc[j] = sum;
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic “central” triad kernel.
 * @param cc       Crypto context
 * @param pk       Public key
 * @param a_enc    Encrypted output array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted input array c
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar
 */
void centralTriadFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  auto tmp       = cc->EvalMult(c_enc[0], scalar_pt);
  auto res       = EvalAddOperation(cc, b_enc[0], tmp);
  #pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; ++j) {
    a_enc[j] = res;
  }
}
