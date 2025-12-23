/**
 * @file RS_SHMEM_OMP_TARGET.h
 * @brief Header file for OpenSHMEM with OpenMP Target offloading implementation
 * of RaiderSTREAM
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights
 * Reserved.
 * @author michael.beebe@ttu.edu
 * @see LICENSE in the top level directory for licensing details
 */

#ifdef _ENABLE_SHMEM_OMP_TARGET_
#ifndef _RS_SHMEM_OMP_TARGET_H_
#define _RS_SHMEM_OMP_TARGET_H_

#include <omp.h>
#include <shmem.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @class RS_SHMEM_OMP_TARGET
 * @brief Implementation of RaiderSTREAM benchmarks using OpenSHMEM with OpenMP
 * Target offloading
 */
class RS_SHMEM_OMP_TARGET : public RSBaseImpl {
private:
  std::string kernelName;  /**< Name of kernel to execute */
  ssize_t streamArraySize; /**< Size of stream arrays */
  ssize_t chunkSize;       /**< Size of local chunk */
  int lArgc;               /**< Local copy of argc */
  char **lArgv;            /**< Local copy of argv */
  int numPEs;              /**< Number of processing elements */ 
  STREAM_TYPE *a;          /**< First stream array */
  STREAM_TYPE *b;          /**< Second stream array */
  STREAM_TYPE *c;          /**< Third stream array */
  STREAM_TYPE *result_a;   /**< First Result array */
  STREAM_TYPE *result_b;   /**< Second Result array */
  STREAM_TYPE *result_c;   /**< Third Result array */
  STREAM_TYPE *d_a;        /**< Device array a */
  STREAM_TYPE *d_b;        /**< Device array b */
  STREAM_TYPE *d_c;        /**< Device array c */
  ssize_t *d_idx1;         /**< Device index array 1 */
  ssize_t *d_idx2;         /**< Device index array 2 */
  ssize_t *d_idx3;         /**< Device index array 3 */
  STREAM_TYPE scalar;      /**< Scalar value for operations */
  int deviceId;            /**< Target device ID */

public:
  /**
   * @brief Constructor
   * @param opts Options for configuring the implementation
   */
  RS_SHMEM_OMP_TARGET(const RSOpts &opts);

  /**
   * @brief Destructor
   */
  ~RS_SHMEM_OMP_TARGET();

  /**
   * @brief Determine local chunk size of PE
   * @param streamArraySize Total size of arrays in problem
   */
  ssize_t getChunkSize(ssize_t streamArraySize);

  /**
   * @brief Allocates data arrays on host and device
   * @return True if allocation successful, false otherwise
   */
  virtual bool allocateData(double * allocTime, double * initTime, 
      double * randomGenTime) override;

  /**
   * @brief Executes the selected benchmark kernel
   * @param[out] TIMES Array to store timing results
   * @param[out] MBPS Array to store bandwidth results
   * @param[out] FLOPS Array to store FLOPS results
   * @param[out] BYTES Array to store bytes transferred
   * @param[out] FLOATOPS Array to store floating point operations
   * @return True if execution successful, false otherwise
   */
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  /**
   * @brief collect all results into one array
   *
   * @param collectTime The time taken to collect all results
   */
  virtual void collectChunks(double * collectTime) override;

  /**
   * @brief Frees allocated memory
   * @return True if deallocation successful, false otherwise
   */
  virtual bool freeData() override;
};

extern "C" { // FIXME: these might need to take in a `int numPEs` argument
/**
 * @brief Sequential copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays per rank
 */
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize);

/**
 * @brief Sequential scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar);

/**
 * @brief Sequential add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays per rank
 */
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize);

/**
 * @brief Sequential triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar);

/**
 * @brief Gather copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Gather index array
 * @param[in] chunkSize Size of arrays per rank
 */
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t chunkSize);

/**
 * @brief Gather scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Gather index array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Gather add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First gather index array
 * @param[in] IDX2 Second gather index array
 * @param[in] chunkSize Size of arrays per rank
 */
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
               ssize_t *IDX2, ssize_t chunkSize);

/**
 * @brief Gather triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First gather index array
 * @param[in] IDX2 Second gather index array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t *IDX2, ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays per rank
 */
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t chunkSize);

/**
 * @brief Scatter scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays per rank
 */
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t chunkSize);

/**
 * @brief Scatter triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] chunkSize Size of arrays per rank
 */
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
            ssize_t *IDX2, ssize_t chunkSize);

/**
 * @brief Scatter-gather scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] IDX3 Third index array
 * @param[in] chunkSize Size of arrays per rank
 */
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
           ssize_t *IDX2, ssize_t *IDX3, ssize_t chunkSize);

/**
 * @brief Scatter-gather triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] IDX3 Third index array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t *IDX3, ssize_t chunkSize,
             STREAM_TYPE scalar);

/**
 * @brief Central copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays per rank
 */
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t chunkSize);

/**
 * @brief Central scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Central add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays per rank
 */
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t chunkSize);

/**
 * @brief Central triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays per rank
 * @param[in] scalar Scaling factor
 */
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar);
}

#endif /* _RS_SHMEM_OMP_TARGET_H_ */
#endif /* _ENABLE_SHMEM_OMP_TARGET_ */
