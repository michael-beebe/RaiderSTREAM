// RS_FHE.cpp

#include "RS_FHE.h"           // The header for this module (declares the functions)
#include "RS_FHE_Config.h"    // Contains STREAM_TYPE and default parameter definitions.
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

#include <iostream>
#include <vector>
#include <cstdlib>

using namespace lbcrypto;
using namespace std;

/// Create a plaintext from a vector of STREAM_TYPE values.
/// For CKKS, this uses MakeCKKSPackedPlaintext; for BFV/BGV, it uses MakePackedPlaintext.
Plaintext CreatePlaintextVector(const CryptoContext<DCRTPoly>& cc, const std::vector<STREAM_TYPE>& values) {
  #if defined(CKKS)
    return cc->MakeCKKSPackedPlaintext(values);
  #else
    return cc->MakePackedPlaintext(values);
  #endif
}

/// Create a plaintext from a single STREAM_TYPE value.
Plaintext CreatePlaintextValue(const CryptoContext<DCRTPoly>& cc, STREAM_TYPE value) {
    std::vector<STREAM_TYPE> values{ value };
    return CreatePlaintextVector(cc, values);
}

/// Encrypt a vector of STREAM_TYPE values.
Ciphertext<DCRTPoly> EncryptVector(const CryptoContext<DCRTPoly>& cc,
                                     const PublicKey<DCRTPoly>& publicKey,
                                     const std::vector<STREAM_TYPE>& values) {
    Plaintext pt = CreatePlaintextVector(cc, values);
    return cc->Encrypt(publicKey, pt);
}

/// Encrypt a single value.
Ciphertext<DCRTPoly> EncryptValue(const CryptoContext<DCRTPoly>& cc,
                                    const PublicKey<DCRTPoly>& publicKey,
                                    STREAM_TYPE value) {
    Plaintext pt = CreatePlaintextValue(cc, value);
    return cc->Encrypt(publicKey, pt);
}

/// Perform homomorphic addition of two ciphertexts.
Ciphertext<DCRTPoly> EvalAddOperation(const CryptoContext<DCRTPoly>& cc,
                                        const Ciphertext<DCRTPoly>& ct1,
                                        const Ciphertext<DCRTPoly>& ct2) {
    return cc->EvalAdd(ct1, ct2);
}

/// Perform homomorphic multiplication of a ciphertext by a plaintext multiplier vector.
/// The multiplier is now provided as a vector of STREAM_TYPE.
Ciphertext<DCRTPoly> EvalMultOperation(const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<STREAM_TYPE>& multiplier) {
#if defined(CKKS)
Plaintext pt = cc->MakeCKKSPackedPlaintext(multiplier);
#else
Plaintext pt = cc->MakePackedPlaintext(multiplier);
#endif
return cc->EvalMult(ct, pt);
}

/// Decrypt a ciphertext and return the resulting vector of numbers.
/// For CKKS, returns real packed values; for BFV/BGV, returns the packed integer values.
std::vector<STREAM_TYPE> DecryptCiphertext(const CryptoContext<DCRTPoly>& cc,
                                        const PrivateKey<DCRTPoly>& secretKey,
                                        const Ciphertext<DCRTPoly>& ct) {
  #if defined(CKKS)
    Plaintext pt = cc->MakeCKKSPackedPlaintext(std::vector<double>());
  #else
    Plaintext pt = cc->MakePackedPlaintext(std::vector<STREAM_TYPE>());
  #endif
    auto decResult = cc->Decrypt(secretKey, ct, &pt);
    if (!decResult.isValid) {
        std::cerr << "Decryption failed!" << std::endl;
        exit(1);
    }
  #if defined(CKKS)
    return pt->GetRealPackedValue();
  #else
    return pt->GetPackedValue();
  #endif
}
