#include "RS_FHE_MPI.h"

// Estimate the size of a ciphertext in bytes
size_t estimateCiphertextSize(const Ciphertext<DCRTPoly>& ct) {
    if (!ct) return 0;
    size_t total = 0;
    const auto& polys = ct->GetElements();
    for (const auto& poly : polys) {
        for (size_t t = 0; t < poly.GetNumOfElements(); ++t) {
            total += poly.GetElementAtIndex(t).GetLength() * sizeof(uint64_t);
        }
    }
    return total;
}

/**
 * @brief Homomorphic "copy" kernel (sequential STREAM copy) for MPI execution.
 * 
 * Copies encrypted data from a_enc to c_enc using OpenMP parallelization
 * within each MPI rank. Each rank processes its local chunks independently.
 * 
 * @param a_enc      Encrypted input array a
 * @param c_enc      Encrypted output array c  
 * @param numChunks  Number of chunks to process (local to this rank)
 */
void seqCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc, 
                    std::vector<Ciphertext<DCRTPoly>> &c_enc, 
                    size_t numChunks) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Simple copy operation: c_enc[chunk_idx] = a_enc[chunk_idx]
        c_enc[chunk_idx] = a_enc[chunk_idx];
        
        // Calculate bytes transferred for this chunk
        size_t bytesTransferred = estimateCiphertextSize(a_enc[chunk_idx]) + 
                                 estimateCiphertextSize(c_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        
        // Debug output (only for first few chunks to avoid spam)
        if (chunk_idx < 3) {
            printf("[DEBUG] seqCopyFHE_MPI: Transferred %zu bytes for chunk %zu\n", 
                   bytesTransferred, chunk_idx);
        }
    }
    
    printf("[DEBUG] seqCopyFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "scale" kernel (sequential STREAM scale) for MPI execution.
 * 
 * Multiplies encrypted data in c_enc by a scalar and stores result in b_enc
 * using OpenMP parallelization within each MPI rank. Each rank processes its 
 * local chunks independently.
 * 
 * @param cc         Crypto context for FHE operations
 * @param b_enc      Encrypted output array b (result = scalar * c_enc)
 * @param c_enc      Encrypted input array c
 * @param numChunks  Number of chunks to process (local to this rank)
 * @param scalar_pt  Precomputed plaintext scalar for multiplication
 */
void seqScaleFHE_MPI(CryptoContext<DCRTPoly> cc,
                     std::vector<Ciphertext<DCRTPoly>> &b_enc,
                     const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                     size_t numChunks, const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // FHE multiplication: b_enc[chunk_idx] = c_enc[chunk_idx] * scalar_pt
        b_enc[chunk_idx] = cc->EvalMult(c_enc[chunk_idx], scalar_pt);
        
        // Calculate bytes transferred for this chunk
        size_t bytesTransferred = estimateCiphertextSize(c_enc[chunk_idx]) + 
                                 estimateCiphertextSize(b_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        
        // Debug output (only for first few chunks to avoid spam)
        if (chunk_idx < 3) {
            printf("[DEBUG] seqScaleFHE_MPI: Transferred %zu bytes for chunk %zu\n", 
                   bytesTransferred, chunk_idx);
        }
    }
    
    printf("[DEBUG] seqScaleFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "add" kernel (sequential STREAM add) for MPI execution.
 * 
 * Adds encrypted data from a_enc and b_enc and stores result in c_enc
 * using OpenMP parallelization within each MPI rank. Each rank processes its 
 * local chunks independently.
 * 
 * @param cc         Crypto context for FHE operations
 * @param a_enc      Encrypted input array a
 * @param b_enc      Encrypted input array b
 * @param c_enc      Encrypted output array c (result = a_enc + b_enc)
 * @param numChunks  Number of chunks to process (local to this rank)
 */
void seqAddFHE_MPI(CryptoContext<DCRTPoly> cc,
                   const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                   const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                   std::vector<Ciphertext<DCRTPoly>> &c_enc,
                   size_t numChunks) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // FHE addition: c_enc[chunk_idx] = a_enc[chunk_idx] + b_enc[chunk_idx]
        c_enc[chunk_idx] = a_enc[chunk_idx] + b_enc[chunk_idx];
        
        // Calculate bytes transferred for this chunk
        size_t bytesTransferred = estimateCiphertextSize(a_enc[chunk_idx]) + 
                                 estimateCiphertextSize(b_enc[chunk_idx]) + 
                                 estimateCiphertextSize(c_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        
        // Debug output (only for first few chunks to avoid spam)
        if (chunk_idx < 3) {
            printf("[DEBUG] seqAddFHE_MPI: Transferred %zu bytes for chunk %zu\n", 
                   bytesTransferred, chunk_idx);
        }
    }
    
    printf("[DEBUG] seqAddFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "triad" kernel (sequential STREAM triad) for MPI execution.
 * 
 * Performs the complex operation: a = b + scalar * c
 * using both FHE multiplication and addition operations.
 * Uses OpenMP parallelization within each MPI rank. Each rank processes its 
 * local chunks independently.
 * 
 * @param cc         Crypto context for FHE operations
 * @param a_enc      Encrypted output array a (result = b_enc + scalar * c_enc)
 * @param b_enc      Encrypted input array b
 * @param c_enc      Encrypted input array c
 * @param numChunks  Number of chunks to process (local to this rank)
 * @param scalar_pt  Precomputed plaintext scalar for multiplication
 */
void seqTriadFHE_MPI(CryptoContext<DCRTPoly> cc,
                     std::vector<Ciphertext<DCRTPoly>> &a_enc,
                     const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                     const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                     size_t numChunks, const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Step 1: FHE multiplication: tmp = c_enc[chunk_idx] * scalar_pt
        auto tmp = cc->EvalMult(c_enc[chunk_idx], scalar_pt);
        
        // Step 2: FHE addition: a_enc[chunk_idx] = b_enc[chunk_idx] + tmp
        a_enc[chunk_idx] = RSFHE::EvalAddOperation(cc, b_enc[chunk_idx], tmp);
        
        // Calculate bytes transferred for this chunk
        size_t bytesTransferred = estimateCiphertextSize(b_enc[chunk_idx]) + 
                                 estimateCiphertextSize(c_enc[chunk_idx]) + 
                                 estimateCiphertextSize(a_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;
        
        // Debug output (only for first few chunks to avoid spam)
        if (chunk_idx < 3) {
            printf("[DEBUG] seqTriadFHE_MPI: Transferred %zu bytes for chunk %zu\n", 
                   bytesTransferred, chunk_idx);
        }
    }
    
    printf("[DEBUG] seqTriadFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "gather-copy" kernel for MPI execution (chunk-based).
 * 
 * Performs gather operations at the ciphertext chunk level to avoid rotation key memory issues.
 * Each MPI rank processes only its local chunks, avoiding cross-rank slot-level rotations.
 * Uses OpenMP for intra-node parallelism.
 * 
 * @param a_enc      Encrypted input array a (source chunks)
 * @param c_enc      Encrypted output array c (destination chunks)
 * @param idx1       Index array for gathering chunk indices
 * @param numChunks  Number of chunks to process (local to this rank)
 */
void gatherCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                       std::vector<Ciphertext<DCRTPoly>> &c_enc,
                       const ssize_t *idx1, size_t numChunks) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level gather: c_enc[chunk_idx] = a_enc[idx1[chunk_idx]]
        // This avoids slot-level rotations by working at ciphertext granularity
        // Each rank only accesses its local chunks, minimizing rotation key requirements
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < a_enc.size()) {
            c_enc[chunk_idx] = a_enc[idx1[chunk_idx]];
            
            // Calculate bytes transferred for this chunk
            size_t bytesTransferred = estimateCiphertextSize(a_enc[idx1[chunk_idx]]) + 
                                     estimateCiphertextSize(c_enc[chunk_idx]);
            bytesTransferredTotal += bytesTransferred;
            
            // Debug output (only for first few chunks to avoid spam)
            if (chunk_idx < 3) {
                printf("[DEBUG] gatherCopyFHE_MPI: Transferred %zu bytes for chunk %zu (idx=%ld)\n", 
                       bytesTransferred, chunk_idx, idx1[chunk_idx]);
            }
        }
    }
    
    printf("[DEBUG] gatherCopyFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "gather-scale" kernel for MPI execution (chunk-based).
 * 
 * Performs gather operations at the ciphertext chunk level to avoid rotation key memory issues.
 * Each MPI rank processes only its local chunks, avoiding cross-rank slot-level rotations.
 * Uses OpenMP for intra-node parallelism.
 * 
 * @param cc         CryptoContext for FHE operations
 * @param b_enc      Encrypted output array b (destination chunks)
 * @param c_enc      Encrypted input array c (source chunks)
 * @param idx1       Index array for gathering chunk indices
 * @param numChunks  Number of chunks to process (local to this rank)
 * @param scalar_pt  Plaintext scalar for multiplication
 */
void gatherScaleFHE_MPI(CryptoContext<DCRTPoly> cc,
                        std::vector<Ciphertext<DCRTPoly>> &b_enc,
                        const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                        const ssize_t *idx1, size_t numChunks,
                        const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level gather scale: b_enc[chunk_idx] = scalar * c_enc[idx1[chunk_idx]]
        // This avoids slot-level rotations by working at ciphertext granularity
        // Each rank only accesses its local chunks, minimizing rotation key requirements
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < c_enc.size()) {
            // FHE multiplication: b_enc[chunk_idx] = c_enc[idx1[chunk_idx]] * scalar_pt
            b_enc[chunk_idx] = cc->EvalMult(c_enc[idx1[chunk_idx]], scalar_pt);
            
            // Calculate bytes transferred for this chunk
            size_t bytesTransferred = estimateCiphertextSize(c_enc[idx1[chunk_idx]]) + 
                                     estimateCiphertextSize(b_enc[chunk_idx]);
            bytesTransferredTotal += bytesTransferred;
            
            // Debug output (only for first few chunks to avoid spam)
            if (chunk_idx < 3) {
                printf("[DEBUG] gatherScaleFHE_MPI: Transferred %zu bytes for chunk %zu (idx=%ld)\n", 
                       bytesTransferred, chunk_idx, idx1[chunk_idx]);
            }
        }
    }
    
    printf("[DEBUG] gatherScaleFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "gather-add" kernel for MPI execution (chunk-based).
 * 
 * Performs gather operations at the ciphertext chunk level to avoid rotation key memory issues.
 * Each MPI rank processes only its local chunks, avoiding cross-rank slot-level rotations.
 * Uses OpenMP for intra-node parallelism.
 * 
 * @param cc         CryptoContext for FHE operations
 * @param a_enc      Encrypted input array a (first source chunks)
 * @param b_enc      Encrypted input array b (second source chunks)
 * @param c_enc      Encrypted output array c (destination chunks)
 * @param idx1       Index array for gathering from a
 * @param idx2       Index array for gathering from b
 * @param numChunks  Number of chunks to process (local to this rank)
 */
void gatherAddFHE_MPI(CryptoContext<DCRTPoly> cc,
                      const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                      const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                      std::vector<Ciphertext<DCRTPoly>> &c_enc,
                      const ssize_t *idx1, const ssize_t *idx2, size_t numChunks) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level gather add: c_enc[chunk_idx] = a_enc[idx1[chunk_idx]] + b_enc[idx2[chunk_idx]]
        // This avoids slot-level rotations by working at ciphertext granularity
        // Each rank only accesses its local chunks, minimizing rotation key requirements
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < a_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < b_enc.size()) {
            
            // FHE addition: c_enc[chunk_idx] = a_enc[idx1[chunk_idx]] + b_enc[idx2[chunk_idx]]
            c_enc[chunk_idx] = RSFHE::EvalAddOperation(cc, a_enc[idx1[chunk_idx]], b_enc[idx2[chunk_idx]]);
            
            // Calculate bytes transferred for this chunk
            size_t bytesTransferred = estimateCiphertextSize(a_enc[idx1[chunk_idx]]) + 
                                     estimateCiphertextSize(b_enc[idx2[chunk_idx]]) + 
                                     estimateCiphertextSize(c_enc[chunk_idx]);
            bytesTransferredTotal += bytesTransferred;
            
            // Debug output (only for first few chunks to avoid spam)
            if (chunk_idx < 3) {
                printf("[DEBUG] gatherAddFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n", 
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }
    
    printf("[DEBUG] gatherAddFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "gather-triad" kernel for MPI execution (chunk-based).
 * 
 * Performs gather operations at the ciphertext chunk level to avoid rotation key memory issues.
 * Each MPI rank processes only its local chunks, avoiding cross-rank slot-level rotations.
 * Uses OpenMP for intra-node parallelism.
 * 
 * @param cc         CryptoContext for FHE operations
 * @param a_enc      Encrypted output array a (destination chunks)
 * @param b_enc      Encrypted input array b (first source chunks)
 * @param c_enc      Encrypted input array c (second source chunks)
 * @param idx1       Index array for gathering from b
 * @param idx2       Index array for gathering from c
 * @param numChunks  Number of chunks to process (local to this rank)
 * @param scalar_pt  Plaintext scalar for multiplication
 */
void gatherTriadFHE_MPI(CryptoContext<DCRTPoly> cc,
                        std::vector<Ciphertext<DCRTPoly>> &a_enc,
                        const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                        const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                        const ssize_t *idx1, const ssize_t *idx2, size_t numChunks,
                        const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level gather triad: a_enc[chunk_idx] = b_enc[idx1[chunk_idx]] + scalar * c_enc[idx2[chunk_idx]]
        // This avoids slot-level rotations by working at ciphertext granularity
        // Each rank only accesses its local chunks, minimizing rotation key requirements
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < b_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < c_enc.size()) {
            
            // Step 1: FHE multiplication: tmp = c_enc[idx2[chunk_idx]] * scalar_pt
            auto tmp = cc->EvalMult(c_enc[idx2[chunk_idx]], scalar_pt);
            
            // Step 2: FHE addition: a_enc[chunk_idx] = b_enc[idx1[chunk_idx]] + tmp
            a_enc[chunk_idx] = RSFHE::EvalAddOperation(cc, b_enc[idx1[chunk_idx]], tmp);
            
            // Calculate bytes transferred for this chunk
            size_t bytesTransferred = estimateCiphertextSize(b_enc[idx1[chunk_idx]]) + 
                                     estimateCiphertextSize(c_enc[idx2[chunk_idx]]) + 
                                     estimateCiphertextSize(a_enc[chunk_idx]);
            bytesTransferredTotal += bytesTransferred;
            
            // Debug output (only for first few chunks to avoid spam)
            if (chunk_idx < 3) {
                printf("[DEBUG] gatherTriadFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n", 
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }
    
    printf("[DEBUG] gatherTriadFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

/**
 * @brief Homomorphic "scatter-copy" kernel for MPI execution (chunk-based).
 * 
 * Performs scatter operations at the ciphertext chunk level to avoid rotation key memory issues.
 * Each MPI rank processes only its local chunks, avoiding cross-rank slot-level rotations.
 * Uses OpenMP for intra-node parallelism.
 * 
 * @param a_enc      Encrypted input array a (source chunks)
 * @param c_enc      Encrypted output array c (destination chunks to scatter to)
 * @param idx1       Index array for scattering chunk indices
 * @param numChunks  Number of chunks to process (local to this rank)
 */
void scatterCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                        std::vector<Ciphertext<DCRTPoly>> &c_enc,
                        const ssize_t *idx1, size_t numChunks) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level scatter: c_enc[idx1[chunk_idx]] = a_enc[chunk_idx]
        // This avoids slot-level rotations by working at ciphertext granularity
        // Each rank only accesses its local chunks, minimizing rotation key requirements
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < c_enc.size()) {
            c_enc[idx1[chunk_idx]] = a_enc[chunk_idx];
            
            // Calculate bytes transferred for this chunk
            size_t bytesTransferred = estimateCiphertextSize(a_enc[chunk_idx]) + 
                                     estimateCiphertextSize(c_enc[idx1[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;
            
            // Debug output (only for first few chunks to avoid spam)
            if (chunk_idx < 3) {
                printf("[DEBUG] scatterCopyFHE_MPI: Transferred %zu bytes for chunk %zu (idx=%ld)\n", 
                       bytesTransferred, chunk_idx, idx1[chunk_idx]);
            }
        }
    }
    
    printf("[DEBUG] scatterCopyFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

// TODO: Add stubs for other kernels as needed
