/**
 * @file RS_SHMEM_OMP_IMPL.c
 * @brief Implementation file for OpenMP+OpenSHMEM STREAM kernels
 * @copyright Copyright (C) 2022-2024 Texas Tech University
 * @author michael.beebe@ttu.edu
 * @license See LICENSE in the top level directory for licensing details
 */

#include <omp.h>
#include <sys/types.h>

/**
 * @brief Copies data from one stream to another using sequential access pattern
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void seqCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[j];
}

/**
 * @brief Scales data in a stream using sequential access pattern
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void seqScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[j] = scalar * c[j];
}

/**
 * @brief Adds data from two streams using sequential access pattern
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void seqAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[j] + b[j];
}

/**
 * @brief Performs triad operation using sequential access pattern
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void seqTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t chunkSize,
              STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**
 * @brief Copies data using gather access pattern
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for gather
 * @param[in] chunkSize Size of arrays
 */
void gatherCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[idx1[j]];
}

/**
 * @brief Scales data using gather access pattern
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] idx1 Index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void gatherScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**
 * @brief Adds data using gather access pattern
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] idx1 First index array for gather
 * @param[in] idx2 Second index array for gather
 * @param[in] chunkSize Size of arrays
 */
void gatherAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
               ssize_t *idx2, ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[idx1[j]] + b[idx2[j]];
}

/**
 * @brief Performs triad operation using gather access pattern
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 First index array for gather
 * @param[in] idx2 Second index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void gatherTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t *idx2, ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**
 * @brief Copies data using scatter access pattern
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 */
void scatterCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                 ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[j];
}

/**
 * @brief Scales data using scatter access pattern
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] idx1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void scatterScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                  ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**
 * @brief Adds data using scatter access pattern
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 */
void scatterAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[j] + b[j];
}

/**
 * @brief Performs triad operation using scatter access pattern
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 Index array for scatter
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void scatterTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
                  ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**
 * @brief Copies data using scatter-gather access pattern
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] idx2 Index array for gather
 * @param[in] chunkSize Size of arrays
 */
void sgCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
            ssize_t *idx2, ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[idx2[j]];
}

/**
 * @brief Scales data using scatter-gather access pattern
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] idx1 Index array for gather
 * @param[in] idx2 Index array for scatter
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void sgScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
             ssize_t *idx2, ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**
 * @brief Adds data using scatter-gather access pattern
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] idx1 Index array for scatter
 * @param[in] idx2 First index array for gather
 * @param[in] idx3 Second index array for gather
 * @param[in] chunkSize Size of arrays
 */
void sgAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
           ssize_t *idx2, ssize_t *idx3, ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
}

/**
 * @brief Performs triad operation using scatter-gather access pattern
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 Index array for gather
 * @param[in] idx2 Index array for scatter
 * @param[in] idx3 Second index array for gather
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void sgTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t *idx1,
             ssize_t *idx2, ssize_t *idx3, ssize_t chunkSize,
             STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**
 * @brief Copies data using central access pattern (single element)
 * @param[in] a Source array
 * @param[in] b Unused array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void centralCopy(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[0] = a[0];
}

/**
 * @brief Scales data using central access pattern (single element)
 * @param[in] a Unused array
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void centralScale(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[0] = scalar * c[0];
}

/**
 * @brief Adds data using central access pattern (single element)
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of arrays
 */
void centralAdd(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t chunkSize) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[0] = a[0] + b[0];
}

/**
 * @brief Performs triad operation using central access pattern (single element)
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of arrays
 * @param[in] scalar Scaling factor
 */
void centralTriad(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t chunkSize, STREAM_TYPE scalar) {
#pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[0] = b[0] + scalar * c[0];
}

/* EOF */
