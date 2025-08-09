#include "RS_FHE_MPI.h"

// Estimate the size of a ciphertext in bytes using serialization.
size_t estimateCiphertextSize(const Ciphertext<DCRTPoly>& ct) {
    if (!ct) return 0;
    std::stringstream ss;
    lbcrypto::Serial::Serialize(ct, ss, lbcrypto::SerType::BINARY);
    return ss.str().length();
}

/**
 * @brief Extracts a single slot from a ciphertext by rotating and masking.
 */
Ciphertext<DCRTPoly> extractSlot(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct, int slot, int batchSize) {
    // Create a mask with 1 at the first slot and 0 elsewhere.
    std::vector<STREAM_TYPE> mask_vec(batchSize, 0);
    mask_vec[0] = 1;
    auto mask_pt = CreatePlaintextVector(cc, mask_vec);

    // Rotate the desired slot to position 0 using EvalRotate
    auto rotated_ct = cc->EvalRotate(ct, -slot);

    // Mask to isolate the value at slot 0
    return cc->EvalMult(rotated_ct, mask_pt);
}

/**
 * @brief Takes a ciphertext with a value at slot 0 and rotates it to a target slot.
 */
Ciphertext<DCRTPoly> insertSlotAtPosition(CryptoContext<DCRTPoly> cc, const Ciphertext<DCRTPoly>& ct, int slot) {
    // Use EvalRotate for composition with power-of-two keys
    return cc->EvalRotate(ct, slot);
}

// ---- Kernel Implementations ----

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
/**
 * @brief Performs a slot-wise gather copy: c[i] = a[idx1[i]].
 * @note The output vector c_enc MUST be initialized with encrypted zeros before calling this kernel.
 */
void gatherCopyFHE_MPI(CryptoContext<DCRTPoly> cc,
                       const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                       std::vector<Ciphertext<DCRTPoly>> &c_enc,
                       const ssize_t *idx1, size_t localStreamSize, int batchSize) {

    #pragma omp parallel for
    for (size_t i = 0; i < localStreamSize; ++i) {
        // 1. Identify source and destination locations
        ssize_t src_global_idx = idx1[i];

        // This kernel only handles intra-rank data movement.
        // A full MPI implementation would require communication.
        // For now, skip any access that would go to another rank.
        if (src_global_idx < 0 || static_cast<size_t>(src_global_idx) >= localStreamSize) {
            continue;
        }

        size_t src_ct_idx = src_global_idx / batchSize;
        int    src_slot_idx = src_global_idx % batchSize;

        size_t dest_ct_idx = i / batchSize;
        int    dest_slot_idx = i % batchSize;

        // Ensure indices are within the bounds of the local rank's data
        if (src_ct_idx >= a_enc.size() || dest_ct_idx >= c_enc.size()) continue;

        // 2. Extract the source slot into a new ciphertext (value is at slot 0)
        Ciphertext<DCRTPoly> extracted_slot_ct = extractSlot(cc, a_enc[src_ct_idx], src_slot_idx, batchSize);

        // 3. Rotate the extracted slot to its destination position
        Ciphertext<DCRTPoly> positioned_slot_ct = insertSlotAtPosition(cc, extracted_slot_ct, dest_slot_idx);

        // 4. Atomically add the result to the destination ciphertext.
        #pragma omp critical
        {
            c_enc[dest_ct_idx] = cc->EvalAdd(c_enc[dest_ct_idx], positioned_slot_ct);
        }
    }
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
void scatterCopyFHE_MPI(CryptoContext<DCRTPoly> cc,
                        const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                        std::vector<Ciphertext<DCRTPoly>> &c_enc,
                        const ssize_t *idx1, size_t localStreamSize, int batchSize) {
    size_t bytesTransferredTotal = 0;
    
    #pragma omp parallel for
    for (size_t i = 0; i < localStreamSize; ++i) {
        // 1. Identify source and destination locations
        size_t src_ct_idx = i / batchSize;
        int    src_slot_idx = i % batchSize;

        ssize_t dest_global_idx = idx1[i];
        size_t dest_ct_idx = dest_global_idx / batchSize;
        int    dest_slot_idx = dest_global_idx % batchSize;

        // Ensure indices are within the bounds of the local rank's data
        if (src_ct_idx >= a_enc.size() || dest_ct_idx >= c_enc.size()) continue;

        // 2. Extract the source slot into a new ciphertext (value is at slot 0)
        Ciphertext<DCRTPoly> extracted_slot_ct = extractSlot(cc, a_enc[src_ct_idx], src_slot_idx, batchSize);

        // 3. Rotate the extracted slot to its destination position
        Ciphertext<DCRTPoly> positioned_slot_ct = insertSlotAtPosition(cc, extracted_slot_ct, dest_slot_idx);

        // 4. Atomically add the result to the destination ciphertext.
        // This is critical because multiple source slots may map to the same destination ciphertext.
        #pragma omp critical
        {
            c_enc[dest_ct_idx] = cc->EvalAdd(c_enc[dest_ct_idx], positioned_slot_ct);
        }
    }
}

/** * @brief Homomorphic "scatter-scale" kernel for MPI execution (chunk-based).
 * 
 * Performs scatter operations at the ciphertext chunk level to avoid rotation key memory issues.
 * Each MPI rank processes only its local chunks, avoiding cross-rank slot-level rotations.
 * Uses OpenMP for intra-node parallelism.
 * 
 * @param cc         CryptoContext for FHE operations
 * @param b_enc      Encrypted output array b (destination chunks to scatter to)
 * @param c_enc      Encrypted input array c (source chunks)
 * @param idx1       Index array for scattering chunk indices
 * @param numChunks  Number of chunks to process (local to this rank)
 * @param scalar_pt  Plaintext scalar for multiplication
 */
void scatterScaleFHE_MPI(CryptoContext<DCRTPoly> cc,
                         std::vector<Ciphertext<DCRTPoly>> &b_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                         const ssize_t *idx1, size_t numChunks,
                         const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level scatter scale: b_enc[idx1[chunk_idx]] = scalar * c_enc[chunk_idx]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < b_enc.size()) {
            b_enc[idx1[chunk_idx]] = cc->EvalMult(c_enc[chunk_idx], scalar_pt);

            size_t bytesTransferred = estimateCiphertextSize(c_enc[chunk_idx]) +
                                     estimateCiphertextSize(b_enc[idx1[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] scatterScaleFHE_MPI: Transferred %zu bytes for chunk %zu (idx=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] scatterScaleFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

void scatterAddFHE_MPI(CryptoContext<DCRTPoly> cc,
                       const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                       const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                       std::vector<Ciphertext<DCRTPoly>> &c_enc,
                       const ssize_t *idx1, const ssize_t *idx2, size_t numChunks) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level scatter add: c_enc[idx1[chunk_idx]] = a_enc[chunk_idx] + b_enc[idx2[chunk_idx]]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < c_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < b_enc.size()) {
            c_enc[idx1[chunk_idx]] = RSFHE::EvalAddOperation(cc, a_enc[chunk_idx], b_enc[idx2[chunk_idx]]);

            size_t bytesTransferred = estimateCiphertextSize(a_enc[chunk_idx]) +
                                     estimateCiphertextSize(b_enc[idx2[chunk_idx]]) +
                                     estimateCiphertextSize(c_enc[idx1[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] scatterAddFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] scatterAddFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

void scatterTriadFHE_MPI(CryptoContext<DCRTPoly> cc,
                         std::vector<Ciphertext<DCRTPoly>> &a_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                         const ssize_t *idx1, const ssize_t *idx2, size_t numChunks,
                         const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Chunk-level scatter triad: a_enc[idx1[chunk_idx]] = b_enc[chunk_idx] + scalar * c_enc[idx2[chunk_idx]]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < a_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < c_enc.size()) {
            // Step 1: FHE multiplication: tmp = c_enc[idx2[chunk_idx]] * scalar_pt
            auto tmp = cc->EvalMult(c_enc[idx2[chunk_idx]], scalar_pt);
            // Step 2: FHE addition: a_enc[idx1[chunk_idx]] = b_enc[chunk_idx] + tmp
            a_enc[idx1[chunk_idx]] = RSFHE::EvalAddOperation(cc, b_enc[chunk_idx], tmp);

            size_t bytesTransferred = estimateCiphertextSize(b_enc[chunk_idx]) +
                                     estimateCiphertextSize(c_enc[idx2[chunk_idx]]) +
                                     estimateCiphertextSize(a_enc[idx1[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] scatterTriadFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] scatterTriadFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}
void sgCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                   std::vector<Ciphertext<DCRTPoly>> &c_enc,
                   const ssize_t *idx1, const ssize_t *idx2, size_t numChunks) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Scatter-gather copy: c_enc[idx2[chunk_idx]] = a_enc[idx1[chunk_idx]]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < a_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < c_enc.size()) {
            c_enc[idx2[chunk_idx]] = a_enc[idx1[chunk_idx]];

            size_t bytesTransferred = estimateCiphertextSize(a_enc[idx1[chunk_idx]]) +
                                     estimateCiphertextSize(c_enc[idx2[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] sgCopyFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] sgCopyFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}
void sgScaleFHE_MPI(CryptoContext<DCRTPoly> cc,
                    std::vector<Ciphertext<DCRTPoly>> &b_enc,
                    const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                    const ssize_t *idx1, const ssize_t *idx2, size_t numChunks,
                    const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Scatter-gather scale: b_enc[idx2[chunk_idx]] = scalar * c_enc[idx1[chunk_idx]]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < c_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < b_enc.size()) {
            b_enc[idx2[chunk_idx]] = cc->EvalMult(c_enc[idx1[chunk_idx]], scalar_pt);

            size_t bytesTransferred = estimateCiphertextSize(c_enc[idx1[chunk_idx]]) +
                                     estimateCiphertextSize(b_enc[idx2[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] sgScaleFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] sgScaleFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

void sgAddFHE_MPI(CryptoContext<DCRTPoly> cc,
                  const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                  const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                  std::vector<Ciphertext<DCRTPoly>> &c_enc,
                  const ssize_t *idx1, const ssize_t *idx2, size_t numChunks) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Scatter-gather add: c_enc[idx2[chunk_idx]] = a_enc[idx1[chunk_idx]] + b_enc[chunk_idx]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < a_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < c_enc.size() &&
            chunk_idx < b_enc.size()) {
            c_enc[idx2[chunk_idx]] = RSFHE::EvalAddOperation(cc, a_enc[idx1[chunk_idx]], b_enc[chunk_idx]);

            size_t bytesTransferred = estimateCiphertextSize(a_enc[idx1[chunk_idx]]) +
                                     estimateCiphertextSize(b_enc[chunk_idx]) +
                                     estimateCiphertextSize(c_enc[idx2[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] sgAddFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] sgAddFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}
void sgTriadFHE_MPI(CryptoContext<DCRTPoly> cc,
                    std::vector<Ciphertext<DCRTPoly>> &a_enc,
                    const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                    const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                    const ssize_t *idx1, const ssize_t *idx2, size_t numChunks,
                    const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // Scatter-gather triad: a_enc[idx2[chunk_idx]] = b_enc[chunk_idx] + scalar * c_enc[idx1[chunk_idx]]
        if (idx1[chunk_idx] >= 0 && static_cast<size_t>(idx1[chunk_idx]) < c_enc.size() &&
            idx2[chunk_idx] >= 0 && static_cast<size_t>(idx2[chunk_idx]) < a_enc.size() &&
            chunk_idx < b_enc.size()) {
            auto tmp = cc->EvalMult(c_enc[idx1[chunk_idx]], scalar_pt);
            a_enc[idx2[chunk_idx]] = RSFHE::EvalAddOperation(cc, b_enc[chunk_idx], tmp);

            size_t bytesTransferred = estimateCiphertextSize(b_enc[chunk_idx]) +
                                     estimateCiphertextSize(c_enc[idx1[chunk_idx]]) +
                                     estimateCiphertextSize(a_enc[idx2[chunk_idx]]);
            bytesTransferredTotal += bytesTransferred;

            if (chunk_idx < 3) {
                printf("[DEBUG] sgTriadFHE_MPI: Transferred %zu bytes for chunk %zu (idx1=%ld, idx2=%ld)\n",
                       bytesTransferred, chunk_idx, idx1[chunk_idx], idx2[chunk_idx]);
            }
        }
    }

    printf("[DEBUG] sgTriadFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}
void centralCopyFHE_MPI(const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                        std::vector<Ciphertext<DCRTPoly>> &c_enc,
                        size_t numChunks) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        c_enc[chunk_idx] = a_enc[0]; // All outputs get the encryption of a[0]
        size_t bytesTransferred = estimateCiphertextSize(a_enc[0]) + estimateCiphertextSize(c_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;

        if (chunk_idx < 3) {
            printf("[DEBUG] centralCopyFHE_MPI: Transferred %zu bytes for chunk %zu\n",
                   bytesTransferred, chunk_idx);
        }
    }

    printf("[DEBUG] centralCopyFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}
void centralScaleFHE_MPI(CryptoContext<DCRTPoly> cc,
                         std::vector<Ciphertext<DCRTPoly>> &b_enc,
                         const std::vector<Ciphertext<DCRTPoly>> &c_enc,
                         size_t numChunks,
                         const lbcrypto::Plaintext &scalar_pt) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // All outputs get the scaled encryption of c[0]
        b_enc[chunk_idx] = cc->EvalMult(c_enc[0], scalar_pt);

        size_t bytesTransferred = estimateCiphertextSize(c_enc[0]) +
                                 estimateCiphertextSize(b_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;

        if (chunk_idx < 3) {
            printf("[DEBUG] centralScaleFHE_MPI: Transferred %zu bytes for chunk %zu\n",
                   bytesTransferred, chunk_idx);
        }
    }

    printf("[DEBUG] centralScaleFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}
void centralAddFHE_MPI(CryptoContext<DCRTPoly> cc,
                       const std::vector<Ciphertext<DCRTPoly>> &a_enc,
                       const std::vector<Ciphertext<DCRTPoly>> &b_enc,
                       std::vector<Ciphertext<DCRTPoly>> &c_enc,
                       size_t numChunks) {
    size_t bytesTransferredTotal = 0;

    #pragma omp parallel for reduction(+:bytesTransferredTotal)
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        // All outputs get the sum of a_enc[0] + b_enc[0]
        c_enc[chunk_idx] = RSFHE::EvalAddOperation(cc, a_enc[0], b_enc[0]);

        size_t bytesTransferred = estimateCiphertextSize(a_enc[0]) +
                                 estimateCiphertextSize(b_enc[0]) +
                                 estimateCiphertextSize(c_enc[chunk_idx]);
        bytesTransferredTotal += bytesTransferred;

        if (chunk_idx < 3) {
            printf("[DEBUG] centralAddFHE_MPI: Transferred %zu bytes for chunk %zu\n",
                   bytesTransferred, chunk_idx);
        }
    }

    printf("[DEBUG] centralAddFHE_MPI: Total bytes transferred: %zu\n", bytesTransferredTotal);
}

// TODO: Add stubs for other kernels as needed
