/**
 * @file RS_OMP_IMPL.c
 * @brief Implementation of OpenMP STREAM benchmark kernels
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#include <omp.h>
#include <sys/types.h>

/**************************************************
 * @brief Copies data from one stream to another.
 *
 * @param a Source array
 * @param b Unused array
 * @param c Destination array
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j];
}

/**************************************************
 * @brief Scales data in a stream.
 *
 * @param a Unused array
 * @param b Destination array
 * @param c Source array
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[j];
}

/**************************************************
 * @brief Adds data from two streams.
 *
 * @param a First source array
 * @param b Second source array
 * @param c Destination array
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
            ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation on stream data.
 *
 * @param a Destination array
 * @param b First source array
 * @param c Second source array
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using gather operation.
 *
 * @param a Source array
 * @param b Unused array
 * @param c Destination array
 * @param idx1 Index array for gather
 * @param streamArraySize Size of the stream array.
 **************************************************/
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]];
}

/**************************************************
 * @brief Scales data using gather operation.
 *
 * @param a Unused array
 * @param b Destination array
 * @param c Source array
 * @param idx1 Index array for gather
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using gather operation.
 *
 * @param a First source array
 * @param b Second source array
 * @param c Destination array
 * @param idx1 First index array for gather
 * @param idx2 Second index array for gather
 * @param streamArraySize Size of the stream array.
 **************************************************/
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
               ssize_t *idx2, ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]] + b[idx2[j]];
}

/**************************************************
 * @brief Performs triad operation using gather.
 *
 * @param a Destination array
 * @param b First source array
 * @param c Second source array
 * @param idx1 First index array for gather
 * @param idx2 Second index array for gather
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t *idx2, ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**************************************************
 * @brief Copies data using scatter operation.
 *
 * @param a Source array
 * @param b Unused array
 * @param c Destination array
 * @param idx1 Index array for scatter
 * @param streamArraySize Size of the stream array.
 **************************************************/
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j];
}

/**************************************************
 * @brief Scales data using scatter operation.
 *
 * @param a Unused array
 * @param b Destination array
 * @param c Source array
 * @param idx1 Index array for scatter
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**************************************************
 * @brief Adds data using scatter operation.
 *
 * @param a First source array
 * @param b Second source array
 * @param c Destination array
 * @param idx1 Index array for scatter
 * @param streamArraySize Size of the stream array.
 **************************************************/
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation using scatter.
 *
 * @param a Destination array
 * @param b First source array
 * @param c Second source array
 * @param idx1 Index array for scatter
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 *
 * @param a Source array
 * @param b Unused array
 * @param c Destination array
 * @param idx1 Index array for scatter
 * @param idx2 Index array for gather
 * @param streamArraySize Size of the stream array.
 **************************************************/
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
            ssize_t *idx2, ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]];
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 *
 * @param a Unused array
 * @param b Destination array
 * @param c Source array
 * @param idx1 Index array for gather
 * @param idx2 Index array for scatter
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
             ssize_t *idx2, ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 *
 * @param a First source array
 * @param b Second source array
 * @param c Destination array
 * @param idx1 Index array for scatter
 * @param idx2 First index array for gather
 * @param idx3 Second index array for gather
 * @param streamArraySize Size of the stream array.
 **************************************************/
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
           ssize_t *idx2, ssize_t *idx3, ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 *
 * @param a Destination array
 * @param b First source array
 * @param c Second source array
 * @param idx1 Index array for gather (c array)
 * @param idx2 Index array for scatter (a array)
 * @param idx3 Index array for gather (b array)
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
             ssize_t *idx2, ssize_t *idx3, ssize_t streamArraySize,
             STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**************************************************
 * @brief Copies data using a central location.
 *
 * @param a Source array
 * @param b Unused array
 * @param c Destination array
 * @param streamArraySize Size of the stream array.
 **************************************************/
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[0] = a[0];
}

/**************************************************
 * @brief Scales data using a central location.
 *
 * @param a Unused array
 * @param b Destination array
 * @param c Source array
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[0] = scalar * c[0];
}

/**************************************************
 * @brief Adds data using a central location.
 *
 * @param a First source array
 * @param b Second source array
 * @param c Destination array
 * @param streamArraySize Size of the stream array.
 **************************************************/
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t streamArraySize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[0] = a[0] + b[0];
}

/**************************************************
 * @brief Performs triad operation using a central location.
 *
 * @param a Destination array
 * @param b First source array
 * @param c Second source array
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[0] = b[0] + scalar * c[0];
}

/* EOF */
