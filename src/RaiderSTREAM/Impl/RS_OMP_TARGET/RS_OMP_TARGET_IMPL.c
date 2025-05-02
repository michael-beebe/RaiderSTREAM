/**
 * @file RS_OMP_TARGET_IMPL.c
 * @brief Implementation of OpenMP target offload STREAM benchmark kernels
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#include <omp.h>
#include <sys/types.h>

#ifndef DO_PRAGMA
#define DO_PRAGMA(x) _Pragma(#x)
#endif

/**
 * @brief Macro to generate OpenMP target offload pragma
 * @details This macro expands to create an OpenMP target teams distribute
 * parallel for simd pragma with device pointer declarations for the passed
 * arguments
 * @note This is manually copied to
 * ../RS_SHMEM_OMP_TARGET/RS_SHMEM_OMP_TARGET_IMPL.c. If you update this,
 * consider updating that file too.
 */
#define LOOP_PRAGMA(...) DO_PRAGMA(omp target teams distribute parallel for simd is_device_ptr(__VA_ARGS__))

/**************************************************
 * @brief Copies data from one stream to another.
 *
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j];
}

/**************************************************
 * @brief Scales data in a stream.
 *
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[j];
}

/**************************************************
 * @brief Adds data from two streams.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
            ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    c[j] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation on stream data.
 *
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using gather operation.
 *
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for gather
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (long j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]];
}

/**************************************************
 * @brief Scales data using gather operation.
 *
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] idx1 Index array for gather
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (long j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using gather operation.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] idx1 First index array for gather
 * @param[in] idx2 Second index array for gather
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
               ssize_t *idx2, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (long j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]] + b[idx2[j]];
}

/**************************************************
 * @brief Performs triad operation using gather.
 *
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 First index array for gather
 * @param[in] idx2 Second index array for gather
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t *idx2, ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (long j = 0; j < streamArraySize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**************************************************
 * @brief Copies data using scatter operation.
 *
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (long j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j];
}

/**************************************************
 * @brief Scales data using scatter operation.
 *
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] idx1 Index array for scatter
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (long j = 0; j < streamArraySize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**************************************************
 * @brief Adds data using scatter operation.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (long j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation using scatter.
 *
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 Index array for scatter
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (long j = 0; j < streamArraySize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 *
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] idx2 Index array for gather
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
            ssize_t *idx2, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (long j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]];
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 *
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] idx1 Index array for gather
 * @param[in] idx2 Index array for scatter
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
             ssize_t *idx2, ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (long j = 0; j < streamArraySize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] idx2 First index array for gather
 * @param[in] idx3 Second index array for gather
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
           ssize_t *idx2, ssize_t *idx3, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1, idx2, idx3)
  for (long j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 *
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 Index array for gather (c array)
 * @param[in] idx2 Index array for scatter (a array)
 * @param[in] idx3 Index array for gather (b array)
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
             ssize_t *idx2, ssize_t *idx3, ssize_t streamArraySize,
             STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c, idx1, idx2, idx3)
  for (long j = 0; j < streamArraySize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**************************************************
 * @brief Copies data using a central location.
 *
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    c[0] = a[0];
}

/**************************************************
 * @brief Scales data using a central location.
 *
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    b[0] = scalar * c[0];
}

/**************************************************
 * @brief Adds data using a central location.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] streamArraySize Size of the stream array
 **************************************************/
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    c[0] = a[0] + b[0];
}

/**************************************************
 * @brief Performs triad operation using a central location.
 *
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] streamArraySize Size of the stream array
 * @param[in] scalar Scalar value for operations
 **************************************************/
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
  LOOP_PRAGMA(a, b, c)
  for (long j = 0; j < streamArraySize; j++)
    a[0] = b[0] + scalar * c[0];
}

/** @cond */ /* Internal documentation */
/* EOF */
/** @endcond */
