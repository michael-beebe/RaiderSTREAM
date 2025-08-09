#ifndef RS_FHE_CONFIG_H
#define RS_FHE_CONFIG_H

// -----------------------------------------------------------------------------
// OpenFHE includes
// -----------------------------------------------------------------------------
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

#if defined(CKKS)
  #include "scheme/ckksrns/ckksrns-ser.h"
#elif defined(BFV)
  #include "scheme/bfvrns/bfvrns-ser.h"
#elif defined(BGV)
  #include "scheme/bgvrns/bgvrns-ser.h"
#endif

#include <iostream>
#include <cstdint>
#include <cstddef>

// -----------------------------------------------------------------------------
// Default FHE Parameters
// -----------------------------------------------------------------------------
static const uint64_t DEFAULT_PTM           = 786433;    // BFV/BGV only
static const int      DEFAULT_DEPTH         = 5;         // multiplicative depth
static const size_t   DEFAULT_RING_DIM      = 32768;     // polynomial ring dimension

#if defined(CKKS)
static const int      DEFAULT_SCALING_MOD_SIZE = 50;      // CKKS only
#endif

// -----------------------------------------------------------------------------
// Chunk‐size: how many slots we’ll pack/encrypt at once.
// By default half the ring (leave headroom for rescaling, etc.).
// -----------------------------------------------------------------------------
static const size_t   DEFAULT_CHUNK_SIZE    = DEFAULT_RING_DIM/2;

using namespace lbcrypto;

// -----------------------------------------------------------------------------
// CryptoContext factory (inline, scheme‑selective)
// -----------------------------------------------------------------------------
inline CryptoContext<DCRTPoly> CreateCryptoContext() {
  CryptoContext<DCRTPoly> cc;

  #ifdef CKKS
      CCParams<CryptoContextCKKSRNS> p;
      p.SetMultiplicativeDepth(DEFAULT_DEPTH);
      p.SetScalingModSize(DEFAULT_SCALING_MOD_SIZE);
      p.SetRingDim(DEFAULT_RING_DIM);
      cc = GenCryptoContext(p);
      std::cout << "[RS_FHE_Config] CKKS context created." << std::endl;
  #elif defined(BFV)
      CCParams<CryptoContextBFVRNS> p;
      p.SetPlaintextModulus(DEFAULT_PTM);
      p.SetMultiplicativeDepth(DEFAULT_DEPTH);
      p.SetRingDim(DEFAULT_RING_DIM);
      cc = GenCryptoContext(p);
      std::cout << "[RS_FHE_Config] BFV context created." << std::endl;
  #elif defined(BGV)
      CCParams<CryptoContextBGVRNS> p;
      p.SetPlaintextModulus(DEFAULT_PTM);
      p.SetMultiplicativeDepth(DEFAULT_DEPTH);
      p.SetRingDim(DEFAULT_RING_DIM);
      cc = GenCryptoContext(p);
      std::cout << "[RS_FHE_Config] BGV context created." << std::endl;
  #else
      #error "You must define one of CKKS, BFV, or BGV when compiling."
  #endif

  // Enable common features for all schemes
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);

  return cc;
}

// -----------------------------------------------------------------------------
// KeyGen helper
// -----------------------------------------------------------------------------

inline KeyPair<DCRTPoly> GenerateKeyPair(
  const CryptoContext<DCRTPoly>& cc) 
{
  std::cout << "[RS_FHE_Config][DEBUG] Entering GenerateKeyPair" << std::endl;

  std::cout << "[RS_FHE_Config][DEBUG] Calling KeyGen()" << std::endl;
  auto kp = cc->KeyGen();
  if (!kp.good()) {
  std::cerr << "[RS_FHE_Config] KeyGen failed!" << std::endl;
  exit(1);
  }
  std::cout << "[RS_FHE_Config][DEBUG] KeyGen successful" << std::endl;

  std::cout << "[RS_FHE_Config][DEBUG] Calling EvalMultKeyGen()" << std::endl;
  cc->EvalMultKeyGen(kp.secretKey);
  std::cout << "[RS_FHE_Config][DEBUG] EvalMultKeyGen successful" << std::endl;

  // // In GenerateKeyPair function, expand the rotation vector
  // std::vector<int32_t> rotations;
  // for (int32_t i = 1; i < 100; ++i) {
  //   rotations.push_back(i);
  // }
  // cc->EvalRotateKeyGen(kp.secretKey, rotations);

  std::cout << "[RS_FHE_Config] Key pair generated." << std::endl;
  return kp;
}

#endif // RS_FHE_CONFIG_H
