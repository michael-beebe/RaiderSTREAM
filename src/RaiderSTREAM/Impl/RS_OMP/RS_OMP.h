/**
 * @file RS_OMP.h
 * @brief Header file for the RS_OMP class implementing OpenMP STREAM benchmarks
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#ifdef _ENABLE_OMP_
#ifndef _RS_OMP_H_
#define _RS_OMP_H_

#include <omp.h>

#include "RaiderSTREAM/RaiderSTREAM.h"

/**
 * @brief RaiderSTREAM OpenMP implementation class
 *
 * This class provides the implementation of the RaiderSTREAM benchmark using
 * OpenMP.
 */
class RS_OMP : public RSBaseImpl {
private:
  std::string kernelName;  /**< Name of the kernel being executed */
  ssize_t streamArraySize; /**< Size of the stream arrays */
  int numPEs;              /**< Number of processing elements */
  int lArgc;               /**< Local argument count */
  char **lArgv;            /**< Local argument vector */
  STREAM_TYPE *a;          /**< First stream array */
  STREAM_TYPE *b;          /**< Second stream array */
  STREAM_TYPE *c;          /**< Third stream array */
  ssize_t *idx1;           /**< First index array */
  ssize_t *idx2;           /**< Second index array */
  ssize_t *idx3;           /**< Third index array */
  STREAM_TYPE scalar;      /**< Scalar value used in operations */

public:
  /**
   * @brief Constructs an RS_OMP object
   * @param opts Configuration options for the benchmark
   */
  RS_OMP(const RSOpts &opts);

  /**
   * @brief Destroys the RS_OMP object
   */
  ~RS_OMP();

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

extern "C" {
/**
 * @brief Sequential copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of arrays to process
 */
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t streamArraySize);

/**
 * @brief Sequential scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Sequential add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of arrays to process
 */
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
            ssize_t streamArraySize);

/**
 * @brief Sequential triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Gather copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for gather
 * @param[in] streamArraySize Size of arrays to process
 */
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t streamArraySize);

/**
 * @brief Gather scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Index array for gather
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Gather add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array for gather
 * @param[in] IDX2 Second index array for gather
 * @param[in] streamArraySize Size of arrays to process
 */
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
               ssize_t *IDX2, ssize_t streamArraySize);

/**
 * @brief Gather triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First index array for gather
 * @param[in] IDX2 Second index array for gather
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t *IDX2, ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Scatter copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for scatter
 * @param[in] streamArraySize Size of arrays to process
 */
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                 ssize_t streamArraySize);

/**
 * @brief Scatter scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 Index array for scatter
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Scatter add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 Index array for scatter
 * @param[in] streamArraySize Size of arrays to process
 */
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                ssize_t streamArraySize);

/**
 * @brief Scatter triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 Index array for scatter
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
                  ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] streamArraySize Size of arrays to process
 */
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
            ssize_t *IDX2, ssize_t streamArraySize);

/**
 * @brief Scatter-gather scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Scatter-gather add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] IDX3 Third index array
 * @param[in] streamArraySize Size of arrays to process
 */
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
           ssize_t *IDX2, ssize_t *IDX3, ssize_t streamArraySize);

/**
 * @brief Scatter-gather triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] IDX1 First index array
 * @param[in] IDX2 Second index array
 * @param[in] IDX3 Third index array
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *IDX1,
             ssize_t *IDX2, ssize_t *IDX3, ssize_t streamArraySize,
             STREAM_TYPE scalar);

/**
 * @brief Central copy operation
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of arrays to process
 */
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t streamArraySize);

/**
 * @brief Central scale operation
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar);

/**
 * @brief Central add operation
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of arrays to process
 */
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t streamArraySize);

/**
 * @brief Central triad operation
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] streamArraySize Size of arrays to process
 * @param[in] scalar Scaling factor
 */
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar);
}

#endif /* _RS_OMP_H_ */
#endif /* _ENABLE_OMP_ */
