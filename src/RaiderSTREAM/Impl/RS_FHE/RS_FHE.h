#ifndef RS_FHE_H
#define RS_FHE_H

#include "RS_FHE_Config.h"  // Defines STREAM_TYPE, default parameters, and includes OpenFHE headers.
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

#include <vector>

using namespace lbcrypto;

namespace RSFHE {

/**
 * @brief Create a plaintext from a vector of STREAM_TYPE values.
 *
 * For CKKS, this function uses MakeCKKSPackedPlaintext;
 * for BFV/BGV, it uses MakePackedPlaintext.
 *
 * @param cc The crypto context.
 * @param values The vector of STREAM_TYPE values.
 * @return Plaintext representing the encoded values.
 */
Plaintext CreatePlaintextVector(const CryptoContext<DCRTPoly>& cc,
                                          const std::vector<STREAM_TYPE>& values);

/**
 * @brief Create a plaintext from a single STREAM_TYPE value.
 *
 * This function wraps the single value into a vector and calls CreatePlaintextVector.
 *
 * @param cc The crypto context.
 * @param value A single STREAM_TYPE value.
 * @return Plaintext encoding the single value.
 */
Plaintext CreatePlaintextValue(const CryptoContext<DCRTPoly>& cc,
                                         STREAM_TYPE value);

/**
 * @brief Encrypt a vector of STREAM_TYPE values.
 *
 * This function first encodes the vector into a plaintext object and then encrypts it.
 *
 * @param cc The crypto context.
 * @param publicKey The public key.
 * @param values The vector of STREAM_TYPE values.
 * @return Ciphertext containing the encrypted vector.
 */
Ciphertext<DCRTPoly> EncryptVector(const CryptoContext<DCRTPoly>& cc,
                                                       const PublicKey<DCRTPoly>& publicKey,
                                                       const std::vector<STREAM_TYPE>& values);

/**
 * @brief Encrypt a single STREAM_TYPE value.
 *
 * This function wraps the single value into a plaintext and then encrypts it.
 *
 * @param cc The crypto context.
 * @param publicKey The public key.
 * @param value A single STREAM_TYPE value.
 * @return Ciphertext containing the encrypted value.
 */
Ciphertext<DCRTPoly> EncryptValue(const CryptoContext<DCRTPoly>& cc,
                                                      const PublicKey<DCRTPoly>& publicKey,
                                                      STREAM_TYPE value);

/**
 * @brief Perform homomorphic addition of two ciphertexts.
 *
 * Uses the EvalAdd function from the crypto context.
 *
 * @param cc The crypto context.
 * @param ct1 First ciphertext operand.
 * @param ct2 Second ciphertext operand.
 * @return Ciphertext containing the result of ct1 + ct2.
 */
Ciphertext<DCRTPoly> EvalAddOperation(const CryptoContext<DCRTPoly>& cc,
                                                          const Ciphertext<DCRTPoly>& ct1,
                                                          const Ciphertext<DCRTPoly>& ct2);

/**
 * @brief Perform homomorphic multiplication of a ciphertext by a plaintext multiplier.
 *
 * The multiplier is provided as a vector of STREAM_TYPE values; for schemes like CKKS,
 * this performs an elementwise multiplication.
 *
 * @param cc The crypto context.
 * @param ct The ciphertext to multiply.
 * @param multiplier The multiplier vector.
 * @return Ciphertext containing the scaled result.
 */
Ciphertext<DCRTPoly> EvalMultOperation(const CryptoContext<DCRTPoly>& cc,
                                                           const Ciphertext<DCRTPoly>& ct,
                                                           const std::vector<STREAM_TYPE>& multiplier);

/**
 * @brief Decrypt a ciphertext and return the resulting vector of numbers.
 *
 * For CKKS, this returns the real packed values;
 * for BFV/BGV, this returns the integer-packed values.
 *
 * @param cc The crypto context.
 * @param secretKey The secret key.
 * @param ct The ciphertext to decrypt.
 * @return Vector of double containing the decrypted values.
 */
std::vector<STREAM_TYPE> DecryptCiphertext(const CryptoContext<DCRTPoly>& cc,
                                        const PrivateKey<DCRTPoly>& secretKey,
                                        const Ciphertext<DCRTPoly>& ct);

} // namespace RSFHE

#endif // RS_FHE_H
