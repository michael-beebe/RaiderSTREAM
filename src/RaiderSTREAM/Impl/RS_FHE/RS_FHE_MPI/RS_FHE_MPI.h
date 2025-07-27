// RS_FHE_MPI.h


#ifndef RS_FHE_MPI_H
#define RS_FHE_MPI_H

#include <vector>
#include <cstddef>
#include <mpi.h>
#include <omp.h>
#include "RSOpts.h"
#include "RSBaseImpl.h"
#include "RS_FHE_Config.h"
#include "openfhe.h"

using namespace lbcrypto;

size_t estimateCiphertextSize(const Ciphertext<DCRTPoly>& ct);

class RS_FHE_MPI : public RSBaseImpl {
public:
  RS_FHE_MPI(const RSOpts &opts);
  ~RS_FHE_MPI();

  bool allocateData() override;
  bool execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) override;
  bool freeData() override;

private:
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
// ... declare all other kernel prototypes here, matching the RS_FHE_OMP.h style, but with _MPI suffix

#endif // RS_FHE_MPI_H
