#include "RS_FHE_MPI.h"

RS_FHE_MPI::RS_FHE_MPI(const RSOpts &opts)
    : RSBaseImpl("RS_FHE_MPI", opts.getKernelTypeFromName(opts.getKernelName())),
      opts(opts),
      kernelName(opts.getKernelName()),
      streamArraySize(opts.getStreamArraySize()),
      numPEs(opts.getNumPEs()),
      idx1(nullptr), idx2(nullptr), idx3(nullptr),
      scalar(3), chunkSize(0) {
    // TODO: MPI-specific initialization
}

RS_FHE_MPI::~RS_FHE_MPI() {
    // TODO: Cleanup
}

bool RS_FHE_MPI::allocateData() {
    std::cout << "[DEBUG] Entering RS_FHE_MPI::allocateData()" << std::endl;
    
    int myRank = -1; /* MPI rank */
    int size = -1;   /* MPI size (number of PEs) */

    if (numPEs == 0) {
        std::cout << "RS_FHE_MPI::allocateData() - ERROR: 'pes' cannot be 0" << std::endl;
        return false;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Calculate the chunk size for each rank */
    chunkSize = streamArraySize / size;
    ssize_t remainder = streamArraySize % size;

    /* Adjust the chunk size for the last process */
    if (myRank == size - 1) {
        chunkSize += remainder;
    }

    std::cout << "[DEBUG] Rank " << myRank << ": chunkSize = " << chunkSize << std::endl;

    // 1) Allocate index arrays for local chunk
    idx1 = new ssize_t[chunkSize];
    if (!idx1) {
        std::cerr << "[ERROR] Memory allocation for idx1 failed!" << std::endl;
        return false;
    }
    idx2 = new ssize_t[chunkSize];
    if (!idx2) {
        std::cerr << "[ERROR] Memory allocation for idx2 failed!" << std::endl;
        return false;
    }
    idx3 = new ssize_t[chunkSize];
    if (!idx3) {
        std::cerr << "[ERROR] Memory allocation for idx3 failed!" << std::endl;
        return false;
    }
    std::cout << "[DEBUG] Rank " << myRank << ": Allocated index arrays" << std::endl;

    // Initialize index arrays
#ifdef _ARRAYGEN_
    initReadIdxArray(idx1, chunkSize, "RaiderSTREAM/arraygen/IDX1.txt");
    initReadIdxArray(idx2, chunkSize, "RaiderSTREAM/arraygen/IDX2.txt");
    initReadIdxArray(idx3, chunkSize, "RaiderSTREAM/arraygen/IDX3.txt");
    std::cout << "[DEBUG] Rank " << myRank << ": Filled index arrays from files" << std::endl;
#else
    initRandomIdxArray(idx1, chunkSize);
    initRandomIdxArray(idx2, chunkSize);
    initRandomIdxArray(idx3, chunkSize);
    std::cout << "[DEBUG] Rank " << myRank << ": Filled index arrays with random values" << std::endl;
#endif

    // 2) Create FHE context & keys (only on rank 0, then broadcast)
    if (myRank == 0) {
        std::cout << "[DEBUG] Creating FHE context..." << std::endl;
        cc = CreateCryptoContext();
        std::cout << "[DEBUG] FHE context created" << std::endl;
        kp = GenerateKeyPair(cc);
        std::cout << "[DEBUG] FHE key pair generated" << std::endl;
    }

    // TODO: Broadcast FHE context and keys to all ranks
    // For now, each rank creates its own context (simpler but less efficient)
    if (myRank != 0) {
        std::cout << "[DEBUG] Rank " << myRank << ": Creating FHE context..." << std::endl;
        cc = CreateCryptoContext();
        kp = GenerateKeyPair(cc);
        std::cout << "[DEBUG] Rank " << myRank << ": FHE context and keys created" << std::endl;
    }

    // 3) Allocate ciphertext buffers for local chunks
    size_t fheChunkSize = DEFAULT_CHUNK_SIZE;
    size_t numChunks = (chunkSize + fheChunkSize - 1) / fheChunkSize;
    a_enc.resize(numChunks);
    b_enc.resize(numChunks);
    c_enc.resize(numChunks);
    std::cout << "[DEBUG] Rank " << myRank << ": Resized ciphertext buffers to numChunks = " << numChunks << std::endl;

    // 4) Initialize and encrypt local data chunks
    std::cout << "[DEBUG] Rank " << myRank << ": Starting chunked batch encryption with chunk size " << fheChunkSize << std::endl;
    
    for (size_t chunk_idx = 0; chunk_idx < numChunks; ++chunk_idx) {
        size_t chunk_start = chunk_idx * fheChunkSize;
        size_t chunk_end = std::min(chunk_start + fheChunkSize, static_cast<size_t>(chunkSize));
        size_t currentChunkSize = chunk_end - chunk_start;

        // Initialize local data arrays
        std::vector<STREAM_TYPE> A(currentChunkSize), B(currentChunkSize), C(currentChunkSize);
        for (size_t i = 0; i < currentChunkSize; ++i) {
            size_t local_idx = chunk_start + i;
            // If using BGV/BFV, values will be reduced mod plaintext modulus automatically
            A[i] = static_cast<STREAM_TYPE>(local_idx % DEFAULT_PTM);
            B[i] = static_cast<STREAM_TYPE>(local_idx % DEFAULT_PTM);
            C[i] = static_cast<STREAM_TYPE>(local_idx % DEFAULT_PTM); 
        }

        // Create packed plaintexts for the chunk
        Plaintext ptA = CreatePlaintextVector(cc, A);
        Plaintext ptB = CreatePlaintextVector(cc, B);
        Plaintext ptC = CreatePlaintextVector(cc, C);

        // Encrypt the packed plaintexts
        a_enc[chunk_idx] = cc->Encrypt(kp.publicKey, ptA);
        b_enc[chunk_idx] = cc->Encrypt(kp.publicKey, ptB);
        c_enc[chunk_idx] = cc->Encrypt(kp.publicKey, ptC);

        std::cout << "[DEBUG] Rank " << myRank << ": Encrypted chunk " << (chunk_idx + 1)
                  << " (" << currentChunkSize << " elements)" << std::endl;

        // DEBUG: decrypt and print first few elements of A vs. decrypted A (only on rank 0)
        if (chunk_idx == 0 && myRank == 0) {
            Plaintext ptA_dec;
            cc->Decrypt(kp.secretKey, a_enc[chunk_idx], &ptA_dec);
            ptA_dec->SetLength(ptA->GetLength());
            #if defined(CKKS)
            auto decA = ptA_dec->GetCKKSPackedValue();
            #else
            auto decA = ptA_dec->GetPackedValue();
            #endif

            std::cout << "[DEBUG] Rank 0 Chunk 0 Plain A[0..9]: ";
            for (size_t i = 0; i < std::min<size_t>(10, A.size()); ++i)
                std::cout << A[i] << ' ';
            std::cout << "\n[DEBUG] Rank 0 Chunk 0 Decr A[0..9]: ";
            for (size_t i = 0; i < std::min<size_t>(10, decA.size()); ++i)
                std::cout << decA[i] << ' ';
            std::cout << std::endl;
        }
    }

    // 5) Set up performance metrics (only on rank 0)
    if (myRank == 0 && !a_enc.empty()) {
        size_t ct_size = estimateCiphertextSize(a_enc[0]);
        std::cout << "[DEBUG] Estimated size of a single ciphertext: " << ct_size << " bytes" << std::endl;

        // Set up BYTES and FLOATOPS arrays for performance calculation
        opts.BYTES[RSBaseImpl::RS_SEQ_COPY]  = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SEQ_SCALE] = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SEQ_ADD]   = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SEQ_TRIAD] = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_GATHER_COPY]  = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_GATHER_SCALE] = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_GATHER_ADD]   = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_GATHER_TRIAD] = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SCATTER_COPY]  = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SCATTER_SCALE] = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SCATTER_ADD]   = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SCATTER_TRIAD] = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SG_COPY]  = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SG_SCALE] = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SG_ADD]   = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_SG_TRIAD] = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_CENTRAL_COPY]  = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_CENTRAL_SCALE] = 2.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_CENTRAL_ADD]   = 3.0 * numChunks * ct_size;
        opts.BYTES[RSBaseImpl::RS_CENTRAL_TRIAD] = 3.0 * numChunks * ct_size;

        opts.FLOATOPS[RSBaseImpl::RS_SEQ_COPY]  = 0.0;
        opts.FLOATOPS[RSBaseImpl::RS_SEQ_SCALE] = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_SEQ_ADD]   = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_SEQ_TRIAD] = 2.0;
        opts.FLOATOPS[RSBaseImpl::RS_GATHER_COPY]  = 0.0;
        opts.FLOATOPS[RSBaseImpl::RS_GATHER_SCALE] = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_GATHER_ADD]   = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_GATHER_TRIAD] = 2.0;
        opts.FLOATOPS[RSBaseImpl::RS_SCATTER_COPY]  = 0.0;
        opts.FLOATOPS[RSBaseImpl::RS_SCATTER_SCALE] = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_SCATTER_ADD]   = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_SCATTER_TRIAD] = 2.0;
        opts.FLOATOPS[RSBaseImpl::RS_SG_COPY]  = 0.0;
        opts.FLOATOPS[RSBaseImpl::RS_SG_SCALE] = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_SG_ADD]   = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_SG_TRIAD] = 2.0;
        opts.FLOATOPS[RSBaseImpl::RS_CENTRAL_COPY]  = 0.0;
        opts.FLOATOPS[RSBaseImpl::RS_CENTRAL_SCALE] = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_CENTRAL_ADD]   = 1.0;
        opts.FLOATOPS[RSBaseImpl::RS_CENTRAL_TRIAD] = 2.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "[DEBUG] Rank " << myRank << ": Finished allocateData()" << std::endl;
    return true;
}

bool RS_FHE_MPI::execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) {
    std::cout << "[DEBUG] Entering RS_FHE_MPI::execute()" << std::endl;
    
    double startTime = 0.0, endTime = 0.0, runTime = 0.0;
    double mbps = 0.0, flops = 0.0;
    double localRunTime = 0.0, localMbps = 0.0, localFlops = 0.0;

    int myRank = -1; /* MPI rank */
    int size = -1;   /* MPI size (number of PEs) */
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate local chunk size (same as in allocateData)
    ssize_t localChunkSize = streamArraySize / size;
    ssize_t remainder = streamArraySize % size;
    if (myRank == size - 1) {
        localChunkSize += remainder;
    }

    // Calculate FHE chunk parameters
    size_t fheChunkSize = DEFAULT_CHUNK_SIZE;
    size_t numChunks = (localChunkSize + fheChunkSize - 1) / fheChunkSize;

    auto kType = getKernelType();
    std::cout << "[DEBUG] Rank " << myRank << ": Kernel type: " << kType << " (" << kernelName << ")" << std::endl;
    std::cout << "[DEBUG] Rank " << myRank << ": localChunkSize = " << localChunkSize << ", numChunks = " << numChunks << std::endl;

    // Handle RS_ALL case by running all kernels
    if (kType == RSBaseImpl::RS_ALL) {
        for (int k = static_cast<int>(RSBaseImpl::RS_SEQ_COPY); k < static_cast<int>(RSBaseImpl::RS_ALL); ++k) {
            RSBaseImpl::RSKernelType currentKernel = static_cast<RSBaseImpl::RSKernelType>(k);
            std::cout << "[DEBUG] Rank " << myRank << ": Running kernel: " << BenchTypeTable[k].Notes << std::endl;
            
            // Run the specific kernel
            if (!executeKernel(currentKernel, TIMES, MBPS, FLOPS, BYTES, FLOATOPS, numChunks, myRank)) {
                std::cerr << "RS_FHE_MPI::execute() - ERROR: failed to execute kernel " << k << std::endl;
                return false;
            }
        }
        return true;
    }

    // Single kernel execution
    return executeKernel(kType, TIMES, MBPS, FLOPS, BYTES, FLOATOPS, numChunks, myRank);
}

bool RS_FHE_MPI::executeKernel(RSBaseImpl::RSKernelType kType, double *TIMES, double *MBPS, double *FLOPS, 
                               double *BYTES, double *FLOATOPS, size_t numChunks, int myRank) {
    double startTime = 0.0, endTime = 0.0, runTime = 0.0;
    double mbps = 0.0, flops = 0.0;
    double localRunTime = 0.0, localMbps = 0.0, localFlops = 0.0;

    // Create scalar plaintext for scaling operations
    auto scalar_pt = CreatePlaintextValue(cc, scalar);

    switch (kType) {
    // ------------------------------
    // SEQUENTIAL KERNELS
    // ------------------------------
    case RSBaseImpl::RS_SEQ_COPY: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling seqCopyFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        seqCopyFHE_MPI(a_enc, c_enc, numChunks);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished seqCopyFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_SEQ_SCALE: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling seqScaleFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        seqScaleFHE_MPI(cc, b_enc, c_enc, numChunks, scalar_pt);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished seqScaleFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_SEQ_ADD: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling seqAddFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        seqAddFHE_MPI(cc, a_enc, b_enc, c_enc, numChunks);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished seqAddFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_SEQ_TRIAD: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling seqTriadFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        seqTriadFHE_MPI(cc, a_enc, b_enc, c_enc, numChunks, scalar_pt);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished seqTriadFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    // ------------------------------
    // GATHER KERNELS (chunk-based to avoid rotation key memory issues)
    // ------------------------------
    case RSBaseImpl::RS_GATHER_COPY: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling gatherCopyFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        gatherCopyFHE_MPI(a_enc, c_enc, idx1, numChunks);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished gatherCopyFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_GATHER_SCALE: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling gatherScaleFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        gatherScaleFHE_MPI(cc, b_enc, c_enc, idx1, numChunks, scalar_pt);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished gatherScaleFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_GATHER_ADD: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling gatherAddFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        gatherAddFHE_MPI(cc, a_enc, b_enc, c_enc, idx1, idx2, numChunks);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished gatherAddFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_GATHER_TRIAD: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling gatherTriadFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        gatherTriadFHE_MPI(cc, a_enc, b_enc, c_enc, idx1, idx2, numChunks, scalar_pt);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished gatherTriadFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    // ------------------------------
    // SCATTER KERNELS (chunk-based to avoid rotation key memory issues)
    // ------------------------------
    case RSBaseImpl::RS_SCATTER_COPY: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling scatterCopyFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        scatterCopyFHE_MPI(a_enc, c_enc, idx1, numChunks);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished scatterCopyFHE_MPI" << std::endl;
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_SCATTER_SCALE: {
        std::cout << "[DEBUG] Rank " << myRank << ": Calling scatterScaleFHE_MPI" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        scatterScaleFHE_MPI(cc, b_enc, c_enc, idx1, numChunks, scalar_pt);
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        std::cout << "[DEBUG] Rank " << myRank << ": Finished scatterScaleFHE_MPI" << std::endl;

        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;
    }

    case RSBaseImpl::RS_SCATTER_ADD:
    case RSBaseImpl::RS_SCATTER_TRIAD:
    // ------------------------------
    // SCATTER-GATHER KERNELS (TODO: Implement)
    // ------------------------------
    case RSBaseImpl::RS_SG_COPY:
    case RSBaseImpl::RS_SG_SCALE:
    case RSBaseImpl::RS_SG_ADD:
    case RSBaseImpl::RS_SG_TRIAD:
    // ------------------------------
    // CENTRAL KERNELS (TODO: Implement)
    // ------------------------------
    case RSBaseImpl::RS_CENTRAL_COPY:
    case RSBaseImpl::RS_CENTRAL_SCALE:
    case RSBaseImpl::RS_CENTRAL_ADD:
    case RSBaseImpl::RS_CENTRAL_TRIAD:
        std::cout << "[DEBUG] Rank " << myRank << ": Kernel type " << kType << " not yet implemented" << std::endl;
        // For now, just measure timing without actual computation
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        // TODO: Implement actual kernel
        MPI_Barrier(MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        
        runTime = calculateRunTime(startTime, endTime);
        mbps = calculateMBPS(opts.BYTES[kType], runTime);
        flops = calculateFLOPS(opts.FLOATOPS[kType], runTime);

        localRunTime = runTime;
        localMbps = mbps;
        localFlops = flops;

        MPI_Reduce(&localRunTime, &TIMES[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMbps, &MBPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFlops, &FLOPS[kType], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        break;

    default:
        std::cerr << "[ERROR] Rank " << myRank << ": Unknown kernel type: " << kType << std::endl;
        return false;
    }

    std::cout << "[DEBUG] Rank " << myRank << ": Kernel " << kType << " completed" << std::endl;
    return true;
}

bool RS_FHE_MPI::freeData() {
    std::cout << "[DEBUG] Entering RS_FHE_MPI::freeData()" << std::endl;
    
    int myRank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    // 1) Free index arrays (using regular delete since we used new[])
    if (idx1) {
        delete[] idx1;
        idx1 = nullptr;
        std::cout << "[DEBUG] Rank " << myRank << ": Freed idx1" << std::endl;
    }
    if (idx2) {
        delete[] idx2;
        idx2 = nullptr;
        std::cout << "[DEBUG] Rank " << myRank << ": Freed idx2" << std::endl;
    }
    if (idx3) {
        delete[] idx3;
        idx3 = nullptr;
        std::cout << "[DEBUG] Rank " << myRank << ": Freed idx3" << std::endl;
    }
    
    // 2) Clear ciphertext vectors (FHE-specific cleanup)
    a_enc.clear();
    b_enc.clear();
    c_enc.clear();
    std::cout << "[DEBUG] Rank " << myRank << ": Cleared ciphertext vectors" << std::endl;
    
    // 3) Reset chunk size
    chunkSize = 0;
    
    // 4) MPI barrier to ensure all ranks complete cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "[DEBUG] Rank " << myRank << ": Finished freeData()" << std::endl;
    return true;
}
