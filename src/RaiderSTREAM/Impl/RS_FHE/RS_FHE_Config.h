#ifndef RS_FHE_CONFIG_H
#define RS_FHE_CONFIG_H

// -----------------------------------------------------------------------------
// Scheme‑Dependent Type Definition
// -----------------------------------------------------------------------------
#if defined(CKKS)
  #define STREAM_TYPE double
#elif defined(BFV) || defined(BGV)
  #define STREAM_TYPE int64_t
#else
  #error "You must define one of CKKS, BFV, or BGV when compiling."
#endif

// -----------------------------------------------------------------------------
// Default FHE Parameters
// -----------------------------------------------------------------------------
static const uint64_t DEFAULT_PTM           = 786433;    // BFV/BGV only
static const int      DEFAULT_DEPTH         = 1;         // multiplicative depth
static const size_t   DEFAULT_RING_DIM      = 65536;     // polynomial ring dimension

#if defined(CKKS)
static const int      DEFAULT_SCALING_MOD_SIZE = 50;      // CKKS only
#endif

// -----------------------------------------------------------------------------
// Chunk‐size: how many slots we’ll pack/encrypt at once.
// By default half the ring (leave headroom for rescaling, etc.).
// -----------------------------------------------------------------------------
static const size_t   DEFAULT_CHUNK_SIZE    = DEFAULT_RING_DIM/2;

// -----------------------------------------------------------------------------
// OpenFHE includes
// -----------------------------------------------------------------------------
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

// -----------------------------------------------------------------------------
// CryptoContext factory (inline, scheme‑selective)
// -----------------------------------------------------------------------------
#ifdef CKKS
 #include "scheme/ckksrns/ckksrns-ser.h"
 inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly> CreateCryptoContext() {
   lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> p;
   p.SetMultiplicativeDepth(DEFAULT_DEPTH)
    .SetScalingModSize(DEFAULT_SCALING_MOD_SIZE)
    .SetRingDim(DEFAULT_RING_DIM);
   auto cc = lbcrypto::GenCryptoContext(p);
   cc->Enable(lbcrypto::ENCRYPTION)
     ->Enable(lbcrypto::LEVELEDSHE)
     ->Enable(lbcrypto::PKE);
   std::cout << "[RS_FHE_Config] CKKS context created." << std::endl;
   return cc;
 }
#endif

#ifdef BFV
 #include "scheme/bfvrns/bfvrns-ser.h"
 inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly> CreateCryptoContext() {
   lbcrypto::CCParams<lbcrypto::CryptoContextBFVRNS> p;
   p.SetPlaintextModulus(DEFAULT_PTM)
    .SetMultiplicativeDepth(DEFAULT_DEPTH)
    .SetRingDim(DEFAULT_RING_DIM);
   auto cc = lbcrypto::GenCryptoContext(p);
   cc->Enable(lbcrypto::ENCRYPTION)
     ->Enable(lbcrypto::KEYSWITCH);
   std::cout << "[RS_FHE_Config] BFV context created." << std::endl;
   return cc;
 }
#endif

#ifdef BGV
 #include "scheme/bgvrns/bgvrns-ser.h"
 inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly> CreateCryptoContext() {
   lbcrypto::CCParams<lbcrypto::CryptoContextBGVRNS> p;
   p.SetPlaintextModulus(DEFAULT_PTM)
    .SetMultiplicativeDepth(DEFAULT_DEPTH)
    .SetRingDim(DEFAULT_RING_DIM);
   auto cc = lbcrypto::GenCryptoContext(p);
   cc->Enable(lbcrypto::ENCRYPTION)
     ->Enable(lbcrypto::KEYSWITCH);
   std::cout << "[RS_FHE_Config] BGV context created." << std::endl;
   return cc;
 }
#endif

// -----------------------------------------------------------------------------
// KeyGen helper
// -----------------------------------------------------------------------------
#include <iostream>
inline lbcrypto::LPKeyPair<lbcrypto::DCRTPoly> GenerateKeyPair(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) 
{
  auto kp = cc->KeyGen();
  if (!kp.good()) {
    std::cerr << "[RS_FHE_Config] KeyGen failed!" << std::endl;
    exit(1);
  }
  cc->EvalMultKeyGen(kp.secretKey);
  cc->EvalSumKeyGen(kp.secretKey);
  std::cout << "[RS_FHE_Config] Key pair generated." << std::endl;
  return kp;
}

#endif // RS_FHE_CONFIG_H
