/**
 * @file RS_SHMEM_OMP.h
 * @brief Header file for the RS_SHMEM_OMP class implementing OpenMP+OpenSHMEM
 * STREAM benchmarks
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#ifdef _ENABLE_SHMEM_OMP_
#ifndef _RS_SHMEM_OMP_H_
#define _RS_SHMEM_OMP_H_

#include <omp.h>
#include <shmem.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief RaiderSTREAM OpenMP+OpenSHMEM implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using
 * OpenMP with OpenSHMEM for distributed memory parallelism.
 */
class RS_SHMEM_OMP : public RSBaseImpl {
private:
  std::string kernelName;  /**< Name of kernel being executed */
  ssize_t streamArraySize; /**< Size of stream arrays */
  int lArgc;               /**< Local argument count */
  char **lArgv;            /**< Local argument vector */
  int numPEs;              /**< Number of processing elements */
  STREAM_TYPE *a;          /**< First stream array */
  STREAM_TYPE *b;          /**< Second stream array */
  STREAM_TYPE *c;          /**< Third stream array */
  ssize_t *idx1;           /**< First index array */
  ssize_t *idx2;           /**< Second index array */
  ssize_t *idx3;           /**< Third index array */
  STREAM_TYPE scalar;      /**< Scalar value used in operations */

public:
  /**
   * @brief Constructs an RS_SHMEM_OMP object
   * @param opts Configuration options for the benchmark
   */
  RS_SHMEM_OMP(const RSOpts &opts);

  /**
   * @brief Destroys the RS_SHMEM_OMP object
   */
  ~RS_SHMEM_OMP();

  /**
   * @brief Allocates and initializes memory for stream arrays
   * @return True if allocation succeeds, false otherwise
   */
  virtual bool allocateData() override;

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
 * @brief Sequential copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize);

/**
 * @brief Sequential scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar);

/**
 * @brief Sequential add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize);

/**
 * @brief Sequential triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar);

/**
 * @brief Gather copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for gather
 * @param[in] chunkSize Size of arrays
 */
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t chunkSize);

/**
 * @brief Gather scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Gather add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array for gather
 * @param[in] IDX2 Second index array for gather
 * @param[in] chunkSize Size of arrays
 */
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
               ssize_t *IDX2, ssize_t chunkSize);

/**
 * @brief Gather triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First index array for gather
 * @param[in] IDX2 Second index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t *IDX2, ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 */
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t chunkSize);

/**
 * @brief Scatter scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 */
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t chunkSize);

/**
 * @brief Scatter triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather copy kernel
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for scatter
 * @param[in] IDX2 Index array for gather
 * @param[in] chunkSize Size of arrays
 */
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
            ssize_t *IDX2, ssize_t chunkSize);

/**
 * @brief Scatter-gather scale kernel
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Index array for scatter
 * @param[in] IDX2 Index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather add kernel
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for scatter
 * @param[in] IDX2 First index array for gather
 * @param[in] IDX3 Second index array for gather
 * @param[in] chunkSize Size of arrays
 */
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
           ssize_t *IDX2, ssize_t *IDX3, ssize_t chunkSize);

/**
 * @brief Scatter-gather triad kernel
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 Index array for gather
 * @param[in] IDX2 Index array for scatter
 * @param[in] IDX3 Second index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t *IDX3, ssize_t chunkSize,
             STREAM_TYPE scalar);

/**
 * @brief Central copy kernel (single element)
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t chunkSize);

/**
 * @brief Central scale kernel (single element)
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar);

/**
 * @brief Central add kernel (single element)
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t chunkSize);

/**
 * @brief Central triad kernel (single element)
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar);
}

#endif /* _RS_SHMEM_OMP_H_ */
#endif /* _ENABLE_SHMEM_OMP_ */
