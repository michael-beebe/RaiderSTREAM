/**
 * @file RS_MPI_OMP_IMPL.c
 * @brief Implementation of RaiderSTREAM kernels using MPI and OpenMP
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights Reserved.
 * @author michael.beebe@ttu.edu
 *
 * This file contains the implementation of various RaiderSTREAM benchmark kernels
 * using a hybrid MPI+OpenMP approach. The kernels include sequential, gather,
 * scatter, scatter-gather and central operations.
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <omp.h>
#include <sys/types.h>

/**
 * @brief Copies data from one stream to another.
 * 
 * This function performs a simple copy operation from array a to array c
 * using OpenMP parallelization.
 *
 * @param[in] a Source array to copy from
 * @param[in] b Unused in this function
 * @param[out] c Destination array to copy to
 * @param[in] chunkSize Size of the arrays to process
 */
void seqCopy(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[j];
}

/**
 * @brief Scales data in a stream.
 * 
 * This function multiplies elements of array c by a scalar value and 
 * stores the result in array b using OpenMP parallelization.
 *
 * @param[in] a Unused in this function
 * @param[out] b Destination array for scaled values
 * @param[in] c Source array to scale
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void seqScale(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[j] = scalar * c[j];
}

/**
 * @brief Adds data from two streams.
 * 
 * This function adds corresponding elements from arrays a and b
 * and stores the result in array c using OpenMP parallelization.
 *
 * @param[in] a First source array for addition
 * @param[in] b Second source array for addition
 * @param[out] c Destination array for sum
 * @param[in] chunkSize Size of the arrays to process
 */
void seqAdd(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[j] + b[j];
}

/**
 * @brief Performs triad operation on stream data.
 * 
 * This function multiplies elements of array c by a scalar and adds
 * corresponding elements from array b, storing results in array a
 * using OpenMP parallelization.
 *
 * @param[out] a Destination array for results
 * @param[in] b First source array for addition
 * @param[in] c Second source array for scaling
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void seqTriad(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**
 * @brief Copies data using gather operation.
 * 
 * This function copies elements from array a to array c using an index array
 * to gather values from non-contiguous locations using OpenMP parallelization.
 *
 * @param[in] a Source array to gather from
 * @param[in] b Unused in this function
 * @param[out] c Destination array
 * @param[in] idx1 Index array for gathering values
 * @param[in] chunkSize Size of the arrays to process
 */
void gatherCopy(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[idx1[j]];
}

/**
 * @brief Scales data using gather operation.
 * 
 * This function scales elements from array c using an index array and stores
 * results in array b using OpenMP parallelization.
 *
 * @param[in] a Unused in this function
 * @param[out] b Destination array
 * @param[in] c Source array to gather from
 * @param[in] idx1 Index array for gathering values
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void gatherScale(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**
 * @brief Adds data using gather operation.
 * 
 * This function adds elements from arrays a and b using index arrays and
 * stores results in array c using OpenMP parallelization.
 *
 * @param[in] a First source array to gather from
 * @param[in] b Second source array to gather from
 * @param[out] c Destination array
 * @param[in] idx1 Index array for gathering from a
 * @param[in] idx2 Index array for gathering from b
 * @param[in] chunkSize Size of the arrays to process
 */
void gatherAdd(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[j] = a[idx1[j]] + b[idx2[j]];
}

/**
 * @brief Performs triad operation using gather.
 * 
 * This function performs a triad operation using index arrays to gather values
 * from arrays b and c using OpenMP parallelization.
 *
 * @param[out] a Destination array
 * @param[in] b First source array to gather from
 * @param[in] c Second source array to gather from
 * @param[in] idx1 Index array for gathering from b
 * @param[in] idx2 Index array for gathering from c
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void gatherTriad(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**
 * @brief Copies data using scatter operation.
 * 
 * This function copies elements from array a to non-contiguous locations in
 * array c using an index array with OpenMP parallelization.
 *
 * @param[in] a Source array
 * @param[in] b Unused in this function
 * @param[out] c Destination array to scatter to
 * @param[in] idx1 Index array for scattering values
 * @param[in] chunkSize Size of the arrays to process
 */
void scatterCopy(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[j];
}

/**
 * @brief Scales data using scatter operation.
 * 
 * This function scales elements from array c and scatters results to
 * array b using an index array with OpenMP parallelization.
 *
 * @param[in] a Unused in this function
 * @param[out] b Destination array to scatter to
 * @param[in] c Source array
 * @param[in] idx1 Index array for scattering values
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void scatterScale(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**
 * @brief Adds data using scatter operation.
 * 
 * This function adds elements from arrays a and b and scatters results to
 * array c using an index array with OpenMP parallelization.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array to scatter to
 * @param[in] idx1 Index array for scattering values
 * @param[in] chunkSize Size of the arrays to process
 */
void scatterAdd(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[j] + b[j];
}

/**
 * @brief Performs triad operation using scatter.
 * 
 * This function performs a triad operation and scatters results to array a
 * using an index array with OpenMP parallelization.
 *
 * @param[out] a Destination array to scatter to
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] idx1 Index array for scattering values
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void scatterTriad(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**
 * @brief Copies data using scatter-gather operation.
 * 
 * This function gathers data from array a and scatters it to array c
 * using index arrays with OpenMP parallelization.
 *
 * @param[in] a Source array to gather from
 * @param[in] b Unused in this function
 * @param[out] c Destination array to scatter to
 * @param[in] idx1 Index array for scattering values
 * @param[in] idx2 Index array for gathering values
 * @param[in] chunkSize Size of the arrays to process
 */
void sgCopy(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[idx2[j]];
}

/**
 * @brief Scales data using scatter-gather operation.
 * 
 * This function gathers data from array c, scales it, and scatters results
 * to array b using index arrays with OpenMP parallelization.
 *
 * @param[in] a Unused in this function
 * @param[out] b Destination array to scatter to
 * @param[in] c Source array to gather from
 * @param[in] idx1 Index array for gathering values
 * @param[in] idx2 Index array for scattering values
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void sgScale(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**
 * @brief Adds data using scatter-gather operation.
 * 
 * This function gathers data from arrays a and b, adds them, and scatters
 * results to array c using index arrays with OpenMP parallelization.
 *
 * @param[in] a First source array to gather from
 * @param[in] b Second source array to gather from
 * @param[out] c Destination array to scatter to
 * @param[in] idx1 Index array for scattering results
 * @param[in] idx2 Index array for gathering from a
 * @param[in] idx3 Index array for gathering from b
 * @param[in] chunkSize Size of the arrays to process
 */
void sgAdd(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
}

/**
 * @brief Performs triad operation using scatter-gather.
 * 
 * This function gathers data from arrays b and c, performs triad operation,
 * and scatters results to array a using index arrays with OpenMP parallelization.
 *
 * @param[out] a Destination array to scatter to
 * @param[in] b First source array to gather from
 * @param[in] c Second source array to gather from
 * @param[in] idx1 Index array for gathering from c
 * @param[in] idx2 Index array for scattering results
 * @param[in] idx3 Index array for gathering from b
 * @param[in] chunkSize Size of the arrays to process
 * @param[in] scalar Value to multiply array elements by
 */
void sgTriad(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**
 * @brief Copies data using a central location.
 * 
 * This function copies a single element from array a to array c
 * using OpenMP parallelization.
 *
 * @param[in] a Source array
 * @param[in] b Unused in this function
 * @param[out] c Destination array
 * @param[in] chunkSize Size of the arrays (unused)
 */
void centralCopy(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[0] = a[0];
}

/**
 * @brief Scales data using a central location.
 * 
 * This function scales a single element from array c and stores it in array b
 * using OpenMP parallelization.
 *
 * @param[in] a Unused in this function
 * @param[out] b Destination array
 * @param[in] c Source array
 * @param[in] chunkSize Size of the arrays (unused)
 * @param[in] scalar Value to multiply array elements by
 */
void centralScale(
  STREAM_TYPE *a,STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    b[0] = scalar * c[0];
}

/**
 * @brief Adds data using a central location.
 * 
 * This function adds single elements from arrays a and b and stores result
 * in array c using OpenMP parallelization.
 *
 * @param[in] a First source array
 * @param[in] b Second source array
 * @param[out] c Destination array
 * @param[in] chunkSize Size of the arrays (unused)
 */
void centralAdd(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    c[0] = a[0] + b[0];
}

/**
 * @brief Performs triad operation using a central location.
 * 
 * This function performs triad operation on single elements from arrays b and c
 * and stores result in array a using OpenMP parallelization.
 *
 * @param[out] a Destination array
 * @param[in] b First source array
 * @param[in] c Second source array
 * @param[in] chunkSize Size of the arrays (unused)
 * @param[in] scalar Value to multiply array elements by
 */
void centralTriad(
  STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
  ssize_t chunkSize, STREAM_TYPE scalar)
{
  #pragma omp parallel for
  for (ssize_t j = 0; j < chunkSize; j++)
    a[0] = b[0] + scalar * c[0];
}

/* EOF */
