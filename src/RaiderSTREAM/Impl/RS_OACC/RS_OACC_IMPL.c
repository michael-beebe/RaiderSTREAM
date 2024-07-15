//
// _RS_OACC_IMPL_C_
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
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[j];
}

/**************************************************
 * @brief Scales data in a stream.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void seqScale(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[j] = scalar * d_c[j];
}

/**************************************************
 * @brief Adds data from two streams.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqAdd(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[j] + d_b[j];
}

/**************************************************
 * @brief Performs triad operation on stream data.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
void seqTriad(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[j] = d_b[j] + scalar * d_c[j];
}

/**************************************************
 * @brief Copies data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void gatherCopy(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[d_idx1[j]];
}

/**************************************************
 * @brief Scales data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void gatherScale(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[j] = scalar * d_c[d_idx1[j]];
}

/**************************************************
 * @brief Adds data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 ************************i**************************/

void gatherAdd(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[d_idx1[j]] + d_b[d_idx2[j]];
}

/**************************************************
 * @brief Performs triad operation using gather.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void gatherTriad(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[j] = d_b[d_idx1[j]] + scalar * d_c[d_idx2[j]];
}

/**************************************************
 * @brief Copies data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void scatterCopy(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[j];
}

/**************************************************
 * @brief Scales data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void scatterScale(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[d_idx1[j]] = scalar * d_c[j];
}

/**************************************************
 * @brief Adds data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void scatterAdd(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[j] + d_b[j];
}

/**************************************************
 * @brief Performs triad operation using scatter.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void scatterTriad(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[d_idx1[j]] = d_b[j] + scalar * d_c[j];
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void sgCopy(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[d_idx2[j]];
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void sgScale(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[d_idx2[j]] = scalar * d_c[d_idx1[j]];
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void sgAdd(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[d_idx2[j]] + d_b[d_idx3[j]];
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void sgTriad(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[d_idx2[j]] = d_b[d_idx3[j]] + scalar * d_c[d_idx1[j]];
}

/**************************************************
 * @brief Copies data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void centralCopy(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[0] = d_a[0];
}

/**************************************************
 * @brief Scales data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void centralScale(
  int ngangs, int nworkers, double *d_a,double *d_b, double *d_c,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[0] = scalar * d_c[0];
}

/**************************************************
 * @brief Adds data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/

void centralAdd(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[0] = d_a[0] + d_b[0];
}

/**************************************************
 * @brief Performs triad operation using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

void centralTriad(
  int ngangs, int nworkers, double *d_a, double *d_b, double *d_c,
  ssize_t streamArraySize, double scalar)
#pragma acc data deviceptr(d_a, d_b, d_c)
#pragma acc parallel num_gangs(ngangs) num_workers(nworkers)
{
  #pragma acc loop
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[0] = d_b[0] + scalar * d_c[0];
}

/* EOF */

