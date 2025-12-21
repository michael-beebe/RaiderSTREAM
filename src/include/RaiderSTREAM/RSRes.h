/**
 * @file RSOpts.h
 * @brief Header file containing the results for RaiderSTREAM
 * benchmark
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights
 * Reserved.
 * @author michael.beebe@ttu.edu
 *
 * See LICENSE in the top level directory for licensing details
 */

#ifndef _RSRES_H_
#define _RSRES_H_

#include <iostream>
#include "RSBaseImpl.h"

#if defined(_ENABLE_SHMEM_OMP_) || defined(_ENABLE_SHMEM_OMP_TARGET_) || defined(_ENABLE_SHMEM_OACC_) || defined(_ENABLE_SHMEM_CUDA_)
#include <shmem.h>
#define CALLOC_DATA(NELEMS, SIZE, TYPE) static_cast<TYPE *>(shmem_calloc(NELEMS, sizeof(TYPE)*SIZE));
#define FREE_DATA(REFERENCE) shmem_free(REFERENCE);
#else
#define CALLOC_DATA(NELEMS, SIZE, TYPE) static_cast<TYPE *>(calloc(NELEMS, sizeof(TYPE) * SIZE));
#define FREE_DATA(REFERENCE) free(REFERENCE);
#endif

/**
 * @brief RSRes class: Container for the results from the RaiderSTREAM benchmark
 */
class RSRes {
  public:
    /**
     * @brief Default constructor
     */
    RSRes ();

    /**
     * @brief Destructor
     */
    ~RSRes ();
    
    /**
     * @brief Storage for the memory bandwidth results in MB/s for each kernel
     */
    double * MBPS;

    /**
     * @brief Storage for the floating point operations per second for each kernel
     */
    double * FLOPS;

    /**
     * @brief Storage for the execution time in seconds for each kernel
     */
    double * TIMES;

    /**
     * @brief Additional Info: Time taken to allocate memory
     */
    double ALLOC_TIME;

    /**
     * @brief Additional Info: Time taken to initalize chunk arrays
     */
    double INIT_TIME;

    /**
     * @brief Additional Info: Time Taken to initalize random arrays
     */
    double RANDOM_GEN_TIME;

    /**
     * @brief Addtional Info: Time taken to collect all results
     */
    double COLLECT_TIME;
};

#endif /* _RSRES_H_ */