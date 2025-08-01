// RS_FHE_MPI.h


#ifndef RS_FHE_MPI_H
#define RS_FHE_MPI_H

#include <vector>
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include "RSOpts.h"
#include "RSBaseImpl.h"
#include "RS_FHE_Config.h"
#include "RS_FHE.h"
#include "openfhe.h"

using namespace lbcrypto;
using RSFHE::CreatePlaintextVector;
using RSFHE::CreatePlaintextValue;

size_t estimateCiphertextSize(const Ciphertext<DCRTPoly>& ct);

class RS_FHE_MPI : public RSBaseImpl {
public:
  RS_FHE_MPI(const RSOpts &opts);
  ~RS_FHE_MPI();

  bool allocateData() override;
  bool execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) override;
  bool freeData() override;

private:
  bool executeKernel(RSBaseImpl::RSKernelType kType, double *TIMES, double *MBPS, double *FLOPS, 
                     double *BYTES, double *FLOATOPS, size_t numChunks, int myRank);
  
  RSOpts opts;
  std::string kernelName;
  ssize_t streamArraySize;
  int numPEs;
  ssize_t *idx1, *idx2, *idx3;
  STREAM_TYPE scalar;
  ssize_t chunkSize;
  CryptoContext<DCRTPoly> cc;
  KeyPair<DCRTPoly> kp;
  std::vector<Ciphertext<DCRTPoly>> a_enc, b_enc, c_enc;
};

// ---- Kernel function prototypes (free functions, not class members) ----
void seqCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc, std::vector<Ciphertext<DCRTPoly>> &c_enc, size_t numChunks);
void seqScaleFHE_MPI(CryptoContext<DCRTPoly> cc, std::vector<Ciphertext<DCRTPoly>> &b_enc, const std::vector<Ciphertext<DCRTPoly>> &c_enc, size_t numChunks, const lbcrypto::Plaintext &scalar_pt);
void seqAddFHE_MPI(CryptoContext<DCRTPoly> cc, const std::vector<Ciphertext<DCRTPoly>> &a_enc, const std::vector<Ciphertext<DCRTPoly>> &b_enc, std::vector<Ciphertext<DCRTPoly>> &c_enc, size_t numChunks);
void seqTriadFHE_MPI(CryptoContext<DCRTPoly> cc, std::vector<Ciphertext<DCRTPoly>> &a_enc, const std::vector<Ciphertext<DCRTPoly>> &b_enc, const std::vector<Ciphertext<DCRTPoly>> &c_enc, size_t numChunks, const lbcrypto::Plaintext &scalar_pt);
void gatherCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc, std::vector<Ciphertext<DCRTPoly>> &c_enc, const ssize_t *idx1, size_t numChunks);
void gatherScaleFHE_MPI(CryptoContext<DCRTPoly> cc, std::vector<Ciphertext<DCRTPoly>> &b_enc, const std::vector<Ciphertext<DCRTPoly>> &c_enc, const ssize_t *idx1, size_t numChunks, const lbcrypto::Plaintext &scalar_pt);
void gatherAddFHE_MPI(CryptoContext<DCRTPoly> cc, const std::vector<Ciphertext<DCRTPoly>> &a_enc, const std::vector<Ciphertext<DCRTPoly>> &b_enc, std::vector<Ciphertext<DCRTPoly>> &c_enc, const ssize_t *idx1, const ssize_t *idx2, size_t numChunks);
void gatherTriadFHE_MPI(CryptoContext<DCRTPoly> cc, std::vector<Ciphertext<DCRTPoly>> &a_enc, const std::vector<Ciphertext<DCRTPoly>> &b_enc, const std::vector<Ciphertext<DCRTPoly>> &c_enc, const ssize_t *idx1, const ssize_t *idx2, size_t numChunks, const lbcrypto::Plaintext &scalar_pt);
void scatterCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc, std::vector<Ciphertext<DCRTPoly>> &c_enc, const ssize_t *idx1, size_t numChunks);
void scatterScaleFHE_MPI(CryptoContext<DCRTPoly> cc,
                         std::vector<Ciphertext<DCRTPoly>> &b_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                         const ssize_t *idx1, size_t numChunks,
                         const lbcrypto::Plaintext &scalar_pt);
void scatterAddFHE_MPI(CryptoContext<DCRTPoly> cc,
                       const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                       const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                       std::vector<Ciphertext<DCRTPoly>> &c_enc,
                       const ssize_t *idx1, const ssize_t *idx2, size_t numChunks);
void scatterTriadFHE_MPI(CryptoContext<DCRTPoly> cc,
                         std::vector<Ciphertext<DCRTPoly>> &a_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                         const ssize_t *idx1, const ssize_t *idx2, size_t numChunks,
                         const lbcrypto::Plaintext &scalar_pt);
// ... declare all other kernel prototypes here, matching the RS_FHE_OMP.h style, but with _MPI suffix

#endif // RS_FHE_MPI_H
