//
// _RS_OMP_IMPL_C_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include <openacc.h>
#include <sys/types.h>

/**************************************************
 * @brief Copies data from one stream to another.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqCopy(
  double *a, double *b, double *c,
  ssize_t streamArraySize)
//#pragma acc data copy(a[:streamArraySize]), copy(c[:streamArraySize])
{
  #pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j];
}

/**************************************************
 * @brief Scales data in a stream.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void seqScale(
  double *a, double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  #pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[j];
}

/**************************************************
 * @brief Adds data from two streams.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqAdd(
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  #pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation on stream data.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void seqTriad(
  double *a, double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  #pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void gatherCopy(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]];
}

/**************************************************
 * @brief Scales data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void gatherScale(
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 ************************i**************************/

void gatherAdd(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]] + b[idx2[j]];
}

/**************************************************
 * @brief Performs triad operation using gather.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void gatherTriad(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**************************************************
 * @brief Copies data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void scatterCopy(
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize)
//#pragma acc data copy(c[:streamArraySize], idx1[:streamArraySize], a[:streamArraySize]) 
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j];
}

/**************************************************
 * @brief Scales data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void scatterScale(
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**************************************************
 * @brief Adds data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void scatterAdd(
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation using scatter.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void scatterTriad(
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void sgCopy(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]];
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void sgScale(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void sgAdd(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
  ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void sgTriad(
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**************************************************
 * @brief Copies data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void centralCopy(
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[0] = a[0];
}

/**************************************************
 * @brief Scales data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void centralScale(
  double *a,double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[0] = scalar * c[0];
}

/**************************************************
 * @brief Adds data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void centralAdd(
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[0] = a[0] + b[0];
}

/**************************************************
 * @brief Performs triad operation using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void centralTriad(
  double *a, double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  //#pragma acc parallel loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[0] = b[0] + scalar * c[0];
}

/* EOF */

