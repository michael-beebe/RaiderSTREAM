//
// _RS_OMP_TARGET_IMPL_C_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include <omp.h>
#include <sys/types.h>

#ifndef ARGS
// cpp stringification shenanigans
//#define DO_PRAGMA_(x) _Pragma(#x)
//#define DO_PRAGMA(x) DO_PRAGMA_(x)

/* Why?
 *
 * We want to pass the array ptrs and the froms
 * as different arguments to the WITH_OFFLOAD macro.
 * But, you can't (easily) escape a comma in cpp directives.
 *
 * So instead, we wrap our ptrs and froms in an enclosing macro,
 * evaluated in place.
 */
#define ARGS(...) __VA_ARGS__
#endif

#ifndef DO_PRAGMA
#define DO_PRAGMA(x) _Pragma(#x)
#endif

/* Below is the OMP offload pragma used for all
 * of this implementation. Modify this to modify all.
 *
 * As a dev-note; you may notice that all invocations of
 * this macro have parens that could probably be removed.
 * I'd suggest against this. The macro becomes significantly
 * less readable, and seems to summon demons when compiled
 * using clang.
 */
#define WITH_OFFLOAD(ptrs, froms) \
  DO_PRAGMA(omp target teams distribute parallel for simd num_teams(nteams) \
                                                          thread_limit(threads) \
                                                          is_device_ptr(ptrs) \
                                                          map(from: froms))


/**************************************************
 * @brief Copies data from one stream to another.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqCopy(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[j];
}

/**************************************************
 * @brief Adds data from two streams.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void seqAdd(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void gatherCopy(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void gatherAdd(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1, idx2), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1, idx2), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**************************************************
 * @brief Copies data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void scatterCopy(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**************************************************
 * @brief Adds data using scatter operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void scatterAdd(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void sgCopy(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1, idx2), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1, idx2), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void sgAdd(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1, idx2, idx3), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c, idx1, idx2, idx3), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**************************************************
 * @brief Copies data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void centralCopy(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a,double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[0] = scalar * c[0];
}

/**************************************************
 * @brief Adds data using a central location.
 * 
 * @param streamArraySize Size of the stream array.
 **************************************************/
void centralAdd(
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize))
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
  int nteams, int threads,
  double *a, double *b, double *c,
  ssize_t streamArraySize, double scalar)
{
  WITH_OFFLOAD(ARGS(a, b, c), ARGS(streamArraySize, scalar))
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[0] = b[0] + scalar * c[0];
}

/* EOF */

