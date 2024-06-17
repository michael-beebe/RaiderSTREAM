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

#ifndef DO_PRAGMA
// cpp stringification shenanigans
#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#endif

// Below is the OMP offload pragma used for all
// of this implementation. Modify this to modify all.
//
// TODO: Multiple GPU support. Investigate `teams` pragma clause.
#define WITH_OFFLOAD(maps) \
  DO_PRAGMA(omp target data maps)

// Same as WITH_OFFLOAD, but for the inner loop after we're on-device.
#define FOR_LOOP_PRAGMA DO_PRAGMA(omp target teams distribute parallel for num_teams(nteams) thread_limit(threads) )


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
  WITH_OFFLOAD(map(from: a[0:streamArraySize]) map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[j];
  }
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
  WITH_OFFLOAD(map(from: c[0:streamArraySize]) map(to: b[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[j] = scalar * c[j];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], b[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[j] + b[j];
  }
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
  WITH_OFFLOAD(map(from: b[0:streamArraySize], c[0:streamArraySize]) \
               map(to: a[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[j] = b[j] + scalar * c[j];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], idx1[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[idx1[j]];
  }
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
  WITH_OFFLOAD(map(from: c[0:streamArraySize], idx1[0:streamArraySize]) map(to: b[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[j] = scalar * c[idx1[j]];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], b[0:streamArraySize], \
                   idx1[0:streamArraySize], idx2[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[idx1[j]] + b[idx2[j]];
  }
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
  WITH_OFFLOAD(map(from: b[0:streamArraySize], c[0:streamArraySize], \
                   idx1[0:streamArraySize], idx2[0:streamArraySize]) \
               map(to: a[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[j] = b[idx1[j]] + scalar * c[idx2[j]];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], idx1[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[j];
  }
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
  WITH_OFFLOAD(map(from: c[0:streamArraySize], idx1[0:streamArraySize]) \
               map(to: b[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[idx1[j]] = scalar * c[j];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], b[0:streamArraySize], idx1[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[j] + b[j];
  }
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
  WITH_OFFLOAD(map(from: b[0:streamArraySize], c[0:streamArraySize], \
                   idx1[0:streamArraySize])                          \
               map(to: a[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[idx1[j]] = b[j] + scalar * c[j];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], idx1[0:streamArraySize], idx2[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[idx2[j]];
  }
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
  WITH_OFFLOAD(map(from: c[0:streamArraySize], idx1[0:streamArraySize], idx2[0:streamArraySize]) \
               map(to: b[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[idx2[j]] = scalar * c[idx1[j]];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], b[0:streamArraySize], \
                   idx1[0:streamArraySize], idx2[0:streamArraySize], idx3[0:streamArraySize]) \
               map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
  }
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
  WITH_OFFLOAD(map(from: b[0:streamArraySize], c[0:streamArraySize], \
                   idx1[0:streamArraySize], idx2[0:streamArraySize], \
                   idx3[0:streamArraySize]) \
               map(to: a[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize]) map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[0] = a[0];
  }
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
  WITH_OFFLOAD(map(from: c[0:streamArraySize]) map(to: b[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[0] = scalar * c[0];
  }
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
  WITH_OFFLOAD(map(from: a[0:streamArraySize], b[0:streamArraySize]) map(to: c[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[0] = a[0] + b[0];
  }
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
  WITH_OFFLOAD(map(from: b[0:streamArraySize], c[0:streamArraySize]) map(to: a[0:streamArraySize]))
  {
    FOR_LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[0] = b[0] + scalar * c[0];
  }
}

/* EOF */

