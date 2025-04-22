#ifndef RS_FHE_H
#define RS_FHE_H

#include "RS_FHE_Config.h"  // Defines STREAM_TYPE, default parameters, and includes OpenFHE headers.
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

#include <vector>

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
lbcrypto::Plaintext CreatePlaintextVector(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
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
lbcrypto::Plaintext CreatePlaintextValue(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
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
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncryptVector(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                       const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
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
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncryptValue(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                      const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
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
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalAddOperation(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                          const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct1,
                                                          const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct2);

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
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMultOperation(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                           const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
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
std::vector<double> DecryptCiphertext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                        const lbcrypto::SecretKey<lbcrypto::DCRTPoly>& secretKey,
                                        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct);

} // namespace RSFHE

#endif // RS_FHE_H
