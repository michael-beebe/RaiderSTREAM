/**
 * @file RS_MPI_OMP.h
 * @brief Header file for the RS_MPI_OMP class implementing hybrid MPI+OpenMP
 * STREAM benchmarks
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#ifdef _ENABLE_MPI_OMP_
#ifndef _RS_MPI_OMP_H_
#define _RS_MPI_OMP_H_

#include <mpi.h>
#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief Class implementing hybrid MPI+OpenMP STREAM benchmarks
 *
 * This class provides an implementation of the STREAM benchmarks using a hybrid
 * MPI+OpenMP approach. It supports sequential, gather, scatter, scatter-gather
 * and central memory access patterns.
 */
class RS_MPI_OMP : public RSBaseImpl {
private:
  std::string kernelName;  /**< Name of the kernel being executed */
  ssize_t streamArraySize; /**< Size of the stream arrays */
  ssize_t chunkSize;       /**< Size of local chunk */
  int lArgc;               /**< Local argument count */
  char **lArgv;            /**< Local argument vector */
  int numPEs;              /**< Number of processing elements */
  STREAM_TYPE *a;          /**< First stream array */
  STREAM_TYPE *b;          /**< Second stream array */
  STREAM_TYPE *c;          /**< Third stream array */
  STREAM_TYPE *result_a;   /**< First collection array */
  STREAM_TYPE *result_b;   /**< Second collection array */
  STREAM_TYPE *result_c;   /**< Third collection array */
  ssize_t *idx1;           /**< First index array */
  ssize_t *idx2;           /**< Second index array */
  ssize_t *idx3;           /**< Third index array */
  STREAM_TYPE scalar;      /**< Scalar value used in operations */

public:
  /**
   * @brief Constructs an RS_MPI_OMP object
   * @param opts Configuration options for the benchmark
   */
  RS_MPI_OMP(const RSOpts &opts);

  /**
   * @brief Destroys the RS_MPI_OMP object
   */
  ~RS_MPI_OMP();

  /**
   * @brief Determine local chunk size of PE
   * @param streamArraySize Total size of arrays in problem
   */
  ssize_t getChunkSize(ssize_t streamArraySize) override;

  /**
   * @brief Allocates and initializes memory for stream arrays
   * @return True if allocation succeeds, false otherwise
   */
  virtual bool allocateData(double *allocTime, double *initTime, double *randomGenTime) override;

  /**
   * @brief collect all results into one array
   *
   * @param collectTime The time taken to collect all results
  **/
  virtual void collectChunks(double * collectTime) override;

  /**
   * @brief Executes the selected benchmark kernel
   * @param[out] TIMES Array to store execution times
   * @param[out] MBPS Array to store bandwidth measurements
   * @param[out] FLOPS Array to store floating point operations per second
   * @param[out] BYTES Array to store bytes transferred
   * @param[out] FLOATOPS Array to store floating point operations performed
   * @return True if execution succeeds, false otherwise
   */
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) override;

  /**
   * @brief Frees allocated memory
   * @return True if deallocation succeeds, false otherwise
   */
  virtual bool freeData() override;
};

extern "C" { // FIXME: these might need to take in a `int numPEs` argument
/**
 * @brief Sequential copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays to process
 */
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize);

/**
 * @brief Sequential scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar);

/**
 * @brief Sequential add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays to process
 */
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize);

/**
 * @brief Sequential triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar);

/**
 * @brief Gather copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Gather index array
 * @param[in] chunkSize Size of arrays to process
 */
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t chunkSize);

/**
 * @brief Gather scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Gather index array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Gather add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First gather index array
 * @param[in] IDX2 Second gather index array
 * @param[in] chunkSize Size of arrays to process
 */
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
               ssize_t *IDX2, ssize_t chunkSize);

/**
 * @brief Gather triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First gather index array
 * @param[in] IDX2 Second gather index array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t *IDX2, ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays to process
 */
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t chunkSize);

/**
 * @brief Scatter scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays to process
 */
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t chunkSize);

/**
 * @brief Scatter triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 Scatter index array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] chunkSize Size of arrays to process
 */
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
            ssize_t *IDX2, ssize_t chunkSize);

/**
 * @brief Scatter-gather scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] IDX3 Third index array
 * @param[in] chunkSize Size of arrays to process
 */
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
           ssize_t *IDX2, ssize_t *IDX3, ssize_t chunkSize);

/**
 * @brief Scatter-gather triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] IDX3 Third index array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t *IDX3, ssize_t chunkSize,
             STREAM_TYPE scalar);

/**
 * @brief Central copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays to process
 */
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t chunkSize);

/**
 * @brief Central scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Central add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays to process
 */
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t chunkSize);

/**
 * @brief Central triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays to process
 * @param[in] scalar Value to multiply by
 */
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar);
}

#endif /* _RS_MPI_OMP_H_ */
#endif /* _ENABLE_MPI_OMP_ */
