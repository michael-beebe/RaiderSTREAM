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
using lbcrypto::Plaintext;
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
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param streamArraySize Number of elements in the arrays
 */
void seqCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, size_t numChunks, ssize_t streamArraySize)
{
    (void)b_enc;  // unused
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
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param streamArraySize Number of elements
 * @param scalar_pt Precomputed plaintext scalar
 */
void seqScaleFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, size_t numChunks, ssize_t streamArraySize, const lbcrypto::Plaintext& scalar_pt)
{
    (void)a_enc;  // unused
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
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param streamArraySize Number of elements
 */
void seqAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, size_t numChunks, ssize_t streamArraySize)
{
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
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param streamArraySize Number of elements
 * @param scalar   Plaintext scalar to multiply
 * @param scalar_pt Precomputed plaintext scalar
 */
void seqTriadFHE(
  CryptoContext<DCRTPoly> cc,
  const PublicKey<DCRTPoly>& pk,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, size_t numChunks, ssize_t streamArraySize, const lbcrypto::Plaintext& scalar_pt)
{
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
 * @param chunkSize Size of each chunk
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param idx1     Index array for gather
 * @param streamArraySize Number of elements
 */
void gatherCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  size_t chunkSize, size_t numChunks, ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < a_enc.size() && j < c_enc.size()) {
        c_enc[j] = a_enc[idx1[j]];
      }
      // else: skip or handle error as needed
    }
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
 * @param chunkSize Size of each chunk
 * @param numChunks Number of chunks (calculated from streamArraySize)
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
  size_t chunkSize, size_t numChunks, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < c_enc.size() && j < b_enc.size()) {
        b_enc[j] = cc->EvalMult(c_enc[idx1[j]], scalar_pt);
      }
      // else: skip or handle error as needed
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize)
{
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < idx2.size() && j < c_enc.size()) {
        ssize_t idx_a = idx1[j];
        ssize_t idx_b = idx2[j];
        if (idx_a >= 0 && static_cast<size_t>(idx_a) < a_enc.size() &&
            idx_b >= 0 && static_cast<size_t>(idx_b) < b_enc.size()) {
          c_enc[j] = EvalAddOperation(cc, a_enc[idx_a], b_enc[idx_b]);
        }
        // else: skip or handle error as needed
      }
      // else: skip or handle error as needed
    }
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
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < idx2.size() && j < a_enc.size()) {
        ssize_t idx_b = idx1[j];
        ssize_t idx_c = idx2[j];
        if (idx_b >= 0 && static_cast<size_t>(idx_b) < b_enc.size() &&
            idx_c >= 0 && static_cast<size_t>(idx_c) < c_enc.size()) {
          auto tmp = cc->EvalMult(c_enc[idx_c], scalar_pt);
          a_enc[j] = EvalAddOperation(cc, b_enc[idx_b], tmp);
        }
        // else: skip or handle error as needed
      }
      // else: skip or handle error as needed
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑copy kernel.
 * @param a_enc    Encrypted input array a
 * @param b_enc Unused
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
 * @param idx1     Index array for scatter
 * @param streamArraySize Number of elements
 */
void scatterCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  size_t chunkSize, ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < a_enc.size() && idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < c_enc.size()) {
        c_enc[static_cast<size_t>(idx1[j])] = a_enc[j];
      }
      // else: skip or handle error as needed
    }
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
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < c_enc.size() && idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < b_enc.size()) {
        b_enc[static_cast<size_t>(idx1[j])] = cc->EvalMult(c_enc[j], scalar_pt);
      }
      // else: skip or handle error as needed
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
 * @param idx1     Index array for scatter
 * @param streamArraySize Number of elements
 */
void scatterAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  size_t chunkSize, ssize_t streamArraySize)
{
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < a_enc.size() && j < b_enc.size() &&
          idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < c_enc.size()) {
        c_enc[static_cast<size_t>(idx1[j])] = EvalAddOperation(cc, a_enc[j], b_enc[j]);
      }
      // else: skip or handle error as needed
    }
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
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < b_enc.size() && j < c_enc.size() &&
          idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < a_enc.size()) {
        auto tmp = cc->EvalMult(c_enc[j], scalar_pt);
        a_enc[static_cast<size_t>(idx1[j])] = EvalAddOperation(cc, b_enc[j], tmp);
      }
      // else: skip or handle error as needed
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑copy kernel.
 * @param a_enc    Encrypted input array a
 * @param b_enc Unused
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize)
{
  (void)b_enc;  // unused
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < idx2.size() &&
          idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < c_enc.size() &&
          idx2[j] >= 0 && static_cast<size_t>(idx2[j]) < a_enc.size()) {
        c_enc[static_cast<size_t>(idx1[j])] = a_enc[static_cast<size_t>(idx2[j])];
      }
      // else: skip or handle error as needed
    }
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
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  (void)a_enc;  // unused
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < idx2.size() &&
          idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < c_enc.size() &&
          idx2[j] >= 0 && static_cast<size_t>(idx2[j]) < b_enc.size()) {
        b_enc[static_cast<size_t>(idx2[j])] = cc->EvalMult(c_enc[static_cast<size_t>(idx1[j])], scalar_pt);
      }
      // else: skip or handle error as needed
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑add kernel.
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize)
{
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < idx2.size() && j < idx3.size()) {
        ssize_t out_idx = idx1[j];
        ssize_t a_idx = idx2[j];
        ssize_t b_idx = idx3[j];
        if (out_idx >= 0 && static_cast<size_t>(out_idx) < c_enc.size() &&
            a_idx >= 0 && static_cast<size_t>(a_idx) < a_enc.size() &&
            b_idx >= 0 && static_cast<size_t>(b_idx) < b_enc.size()) {
          c_enc[static_cast<size_t>(out_idx)] = EvalAddOperation(cc, a_enc[static_cast<size_t>(a_idx)], b_enc[static_cast<size_t>(b_idx)]);
        }
        // else: skip or handle error as needed
      }
      // else: skip or handle error as needed
    }
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
 * @param chunkSize Size of each chunk
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
  size_t chunkSize, ssize_t streamArraySize,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
  #pragma omp parallel for
  for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
    size_t start = chunk_idx * chunkSize;
    size_t end = std::min(start + chunkSize, static_cast<size_t>(streamArraySize));
    for (size_t j = start; j < end; ++j) {
      if (j < idx1.size() && j < idx2.size() && j < idx3.size() &&
          idx1[j] >= 0 && static_cast<size_t>(idx1[j]) < c_enc.size() &&
          idx2[j] >= 0 && static_cast<size_t>(idx2[j]) < a_enc.size() &&
          idx3[j] >= 0 && static_cast<size_t>(idx3[j]) < b_enc.size()) {
        auto tmp = cc->EvalMult(c_enc[static_cast<size_t>(idx1[j])], scalar_pt);
        a_enc[static_cast<size_t>(idx2[j])] = EvalAddOperation(cc, b_enc[static_cast<size_t>(idx3[j])], tmp);
      }
      // else: skip or handle error as needed
    }
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
  const std::vector<Ciphertext<DCRTPoly>>& /*b_enc*/,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize)
{
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = a_enc[0]; // Central: use only the first chunk
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
  const PublicKey<DCRTPoly>& /*pk*/,
  const std::vector<Ciphertext<DCRTPoly>>& /*a_enc*/,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize, STREAM_TYPE scalar)
{
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    auto scalar_pt = CreatePlaintextValue(cc, scalar);
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        b_enc[chunk_idx] = cc->EvalMult(c_enc[0], scalar_pt); // Central: use only the first chunk
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
  size_t chunkSize, ssize_t streamArraySize)
{
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = EvalAddOperation(cc, a_enc[0], b_enc[0]); // Central: use only the first chunk
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
  const PublicKey<DCRTPoly>& /*pk*/,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t chunkSize, ssize_t streamArraySize, STREAM_TYPE scalar)
{
    size_t numChunks = (streamArraySize + chunkSize - 1) / chunkSize;
    auto scalar_pt = CreatePlaintextValue(cc, scalar);
    #pragma omp parallel for
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        auto tmp = cc->EvalMult(c_enc[0], scalar_pt); // Central: use only the first chunk
        a_enc[chunk_idx] = EvalAddOperation(cc, b_enc[0], tmp);
    }
}
