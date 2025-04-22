// RS_FHE_OMP.h
// ------------------------------
// OpenFHE‑backed OpenMP implementation of RaiderSTREAM.
//
//
// To enable, define _ENABLE_FHE_OMP_

#ifdef _ENABLE_FHE_OMP_
#ifndef _RS_FHE_OMP_H_
#define _RS_FHE_OMP_H_

#include <omp.h>
#include <string>
#include <vector>

#include "RS_FHE.h" // core plaintext/encrypt/decrypt/Eval wrappers
#include "RS_FHE_Config.h" // STREAM_TYPE, default params, CreateCryptoContext(), GenerateKeyPair()
#include "RaiderSTREAM/RaiderSTREAM.h" // pulls in RSBaseImpl, RSOpts, STREAM_TYPE

// OpenFHE types
using lbcrypto::Ciphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::LPKeyPair;

class RS_FHE_OMP : public RSBaseImpl {
private:
  std::string kernelName;
  ssize_t streamArraySize;
  STREAM_TYPE scalar;

  // FHE machinery
  CryptoContext<DCRTPoly> cryptoContext;
  LPKeyPair<DCRTPoly> keyPair;

  // Encrypted data arrays
  std::vector<Ciphertext<DCRTPoly>> a_enc, b_enc, c_enc;

  // Index arrays (unchanged)
  std::vector<ssize_t> idx1, idx2, idx3;

public:
  explicit RS_FHE_OMP(const RSOpts &opts);
  ~RS_FHE_OMP() override;

  bool allocateData() override;
  bool execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES,
               double *FLOATOPS) override;
  bool freeData() override;
};

#endif // _RS_FHE_OMP_H_
#endif // _ENABLE_FHE_OMP_
