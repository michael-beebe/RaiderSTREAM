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
 * @param c_enc    Encrypted output array c
 * @param numChunks Number of chunks (calculated from streamArraySize)
 */
void seqCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t numChunks)
{
    size_t bytesTransferredTotal = 0;
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = a_enc[chunk_idx];
        size_t bytesTransferred = estimateCiphertextSize(a_enc[chunk_idx]) + estimateCiphertextSize(c_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        printf("Transferred %zu bytes for chunk %zu\n", bytesTransferred, chunk_idx);
    }
    printf("Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic “scale” kernel (sequential STREAM scale).
 * @param cc       Crypto context
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param scalar_pt Precomputed plaintext scalar
 */
void seqScaleFHE(
  CryptoContext<DCRTPoly> cc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t numChunks, const lbcrypto::Plaintext& scalar_pt)
{
    size_t bytesTransferredTotal = 0;
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        b_enc[chunk_idx] = cc->EvalMult(c_enc[chunk_idx], scalar_pt);
        size_t bytesTransferred =
            estimateCiphertextSize(c_enc[chunk_idx]) +
            estimateCiphertextSize(b_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        printf("Transferred %zu bytes for chunk %zu\n", bytesTransferred, chunk_idx);
    }
    printf("Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic “add” kernel (sequential STREAM add).
 * @param cc       Crypto context
 * @param a_enc    Encrypted input array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted output array c
 * @param numChunks Number of chunks (calculated from streamArraySize)
 */
void seqAddFHE(
  CryptoContext<DCRTPoly> cc,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t numChunks)
{
    size_t bytesTransferredTotal = 0;
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = a_enc[chunk_idx] + b_enc[chunk_idx];
        size_t bytesTransferred =
            estimateCiphertextSize(a_enc[chunk_idx]) +
            estimateCiphertextSize(b_enc[chunk_idx]) +
            estimateCiphertextSize(c_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        printf("Transferred %zu bytes for chunk %zu\n", bytesTransferred, chunk_idx);
    }
    printf("Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic “triad” kernel (sequential STREAM triad).
 *        Computes a = b + scalar * c.
 * @param cc       Crypto context
 * @param a_enc    Encrypted output array a
 * @param b_enc    Encrypted input array b
 * @param c_enc    Encrypted input array c
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param scalar_pt Precomputed plaintext scalar
 */
void seqTriadFHE(
  CryptoContext<DCRTPoly> cc,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  size_t numChunks, const lbcrypto::Plaintext& scalar_pt)
{
    size_t bytesTransferredTotal = 0;
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        auto tmp = cc->EvalMult(c_enc[chunk_idx], scalar_pt);
        a_enc[chunk_idx] = EvalAddOperation(cc, b_enc[chunk_idx], tmp);
        size_t bytesTransferred =
            estimateCiphertextSize(b_enc[chunk_idx]) +
            estimateCiphertextSize(c_enc[chunk_idx]) +
            estimateCiphertextSize(a_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        printf("Transferred %zu bytes for chunk %zu\n", bytesTransferred, chunk_idx);
    }
    printf("Total bytes transferred: %zu\n", bytesTransferredTotal);
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑copy kernel.
 * @param a_enc    Encrypted input array a
 * @param c_enc    Encrypted output array c
 * @param chunkSize Size of each chunk
 * @param numChunks Number of chunks (calculated from streamArraySize)
 * @param idx1     Index array for gather
 * @param streamArraySize Number of elements
 */
void gatherCopyFHE(
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  size_t /*chunkSize*/, size_t numChunks, ssize_t /*streamArraySize*/)
{
  // Expect idx1.size() == numChunks (chunk-level mapping).
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= c_enc.size()) continue;
    if (k >= idx1.size())  continue;

    const ssize_t s = idx1[k];
    if (s < 0) continue;

    const size_t src = static_cast<size_t>(s);
    if (src < a_enc.size()) {
      c_enc[k] = a_enc[src];
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑scale kernel.
 * @param cc       Crypto context
 * @param b_enc    Encrypted output array b
 * @param c_enc    Encrypted input array c
 * @param chunkSize Size of each chunk
 * @param idx1     Index array for gather
 * @param streamArraySize Number of elements
 * @param scalar_pt Precomputed plaintext scalar
 */
void gatherScaleFHE(
    CryptoContext<DCRTPoly> cc,
    std::vector<Ciphertext<DCRTPoly>>& b_enc,
    const std::vector<Ciphertext<DCRTPoly>>& c_enc,
    const std::vector<ssize_t>& idx1,
    size_t /*chunkSize*/, size_t numChunks, ssize_t /*streamArraySize*/,
    const lbcrypto::Plaintext& scalar_pt)
{
  // Expect idx1.size() == numChunks (chunk-level mapping).
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= b_enc.size() || k >= idx1.size()) continue;

    const ssize_t s = idx1[k];
    if (s < 0) continue; // skip invalid mapping

    const size_t src = static_cast<size_t>(s);
    if (src < c_enc.size()) {
      b_enc[k] = cc->EvalMult(c_enc[src], scalar_pt);
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
  size_t /*chunkSize*/, ssize_t /*streamArraySize*/)
{
  const size_t numChunks = c_enc.size();
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= idx1.size() || k >= idx2.size()) continue;

    const ssize_t s1 = idx1[k];
    const ssize_t s2 = idx2[k];
    if (s1 < 0 || s2 < 0) continue; // skip invalid mappings

    const size_t sa = static_cast<size_t>(s1);
    const size_t sb = static_cast<size_t>(s2);
    if (sa < a_enc.size() && sb < b_enc.size()) {
      c_enc[k] = EvalAddOperation(cc, a_enc[sa], b_enc[sb]);
    }
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic gather‑triad kernel.
 * @param cc       Crypto context
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
  const PublicKey<DCRTPoly>& /*pk*/,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  size_t /*chunkSize*/, ssize_t /*streamArraySize*/,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  const size_t numChunks = a_enc.size();

  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= idx1.size() || k >= idx2.size()) continue;

    const ssize_t sb_i = idx1[k];
    const ssize_t sc_i = idx2[k];
    if (sb_i < 0 || sc_i < 0) continue; // skip invalid mappings

    const size_t sb = static_cast<size_t>(sb_i);
    const size_t sc = static_cast<size_t>(sc_i);
    if (sb < b_enc.size() && sc < c_enc.size()) {
      auto tmp = cc->EvalMult(c_enc[sc], scalar_pt);
      a_enc[k] = EvalAddOperation(cc, b_enc[sb], tmp);
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
  const std::vector<Ciphertext<DCRTPoly>>& /*b_enc*/,
  std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  size_t /*chunkSize*/, ssize_t /*streamArraySize*/)
{
  const size_t numChunks = c_enc.size();
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= idx1.size() || k >= idx2.size()) continue;
    const ssize_t d = idx1[k], s = idx2[k];
    if (d < 0 || s < 0) continue;
    const size_t dst = static_cast<size_t>(d);
    const size_t src = static_cast<size_t>(s);
    if (dst < c_enc.size() && src < a_enc.size())
      c_enc[dst] = a_enc[src];
  }
}

// -----------------------------------------------------------------------------
/**
 * @brief Homomorphic scatter‑gather‑scale kernel.
 * @param cc       Crypto context
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
  const PublicKey<DCRTPoly>& /*pk*/,
  const std::vector<Ciphertext<DCRTPoly>>& a_enc,
  std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  size_t /*chunkSize*/, ssize_t /*streamArraySize*/,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  const size_t numChunks = b_enc.size();
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= idx1.size() || k >= idx2.size()) continue;
    const ssize_t s = idx1[k], d = idx2[k];
    if (s < 0 || d < 0) continue;
    const size_t src = static_cast<size_t>(s);
    const size_t dst = static_cast<size_t>(d);
    if (src < c_enc.size() && dst < b_enc.size())
      b_enc[dst] = cc->EvalMult(c_enc[src], scalar_pt);
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
  size_t /*chunkSize*/, ssize_t /*streamArraySize*/)
{
  const size_t numChunks = c_enc.size();
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= idx1.size() || k >= idx2.size() || k >= idx3.size()) continue;
    const ssize_t d = idx1[k], sa = idx2[k], sb = idx3[k];
    if (d < 0 || sa < 0 || sb < 0) continue;
    const size_t dst = static_cast<size_t>(d);
    const size_t ia  = static_cast<size_t>(sa);
    const size_t ib  = static_cast<size_t>(sb);
    if (dst < c_enc.size() && ia < a_enc.size() && ib < b_enc.size())
      c_enc[dst] = EvalAddOperation(cc, a_enc[ia], b_enc[ib]);
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
  const PublicKey<DCRTPoly>& /*pk*/,
  std::vector<Ciphertext<DCRTPoly>>& a_enc,
  const std::vector<Ciphertext<DCRTPoly>>& b_enc,
  const std::vector<Ciphertext<DCRTPoly>>& c_enc,
  const std::vector<ssize_t>& idx1,
  const std::vector<ssize_t>& idx2,
  const std::vector<ssize_t>& idx3,
  size_t /*chunkSize*/, ssize_t /*streamArraySize*/,
  STREAM_TYPE scalar)
{
  auto scalar_pt = CreatePlaintextValue(cc, scalar);
  const size_t numChunks = a_enc.size();
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < numChunks; ++k) {
    if (k >= idx1.size() || k >= idx2.size() || k >= idx3.size()) continue;
    const ssize_t sc = idx1[k], d = idx2[k], sb = idx3[k];
    if (sc < 0 || d < 0 || sb < 0) continue;
    const size_t ic  = static_cast<size_t>(sc);
    const size_t dst = static_cast<size_t>(d);
    const size_t ib  = static_cast<size_t>(sb);
    if (ic < c_enc.size() && ib < b_enc.size() && dst < a_enc.size()) {
      auto tmp = cc->EvalMult(c_enc[ic], scalar_pt);
      a_enc[dst] = EvalAddOperation(cc, b_enc[ib], tmp);
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
