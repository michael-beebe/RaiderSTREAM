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
#include <sys/time.h>
#include <sys/types.h>

double mySecond() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/**************************************************
 * @brief Copies data from one stream to another.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/
double seqCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
               ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Scales data in a stream.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
double seqScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                ssize_t streamArraySize, STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[j] = scalar * d_c[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Adds data from two streams.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/
double seqAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
              ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[j] + d_b[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Performs triad operation on stream data.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/
double seqTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
                ssize_t streamArraySize, STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[j] = d_b[j] + scalar * d_c[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Copies data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double gatherCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                  STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1)
  for (ssize_t j = 0; j < streamArraySize; j++) {
    d_c[j] = d_a[d_idx1[j]];
  }
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Scales data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double gatherScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t streamArraySize,
                   STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[j] = scalar * d_c[d_idx1[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Adds data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 ************************i**************************/

double gatherAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                 STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t *d_idx2,
                 ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[j] = d_a[d_idx1[j]] + d_b[d_idx2[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Performs triad operation using gather.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double gatherTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t *d_idx2,
                   ssize_t streamArraySize, STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[j] = d_b[d_idx1[j]] + scalar * d_c[d_idx2[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Copies data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double scatterCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Scales data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double scatterScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t streamArraySize,
                    STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[d_idx1[j]] = scalar * d_c[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Adds data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double scatterAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                  STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[j] + d_b[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Performs triad operation using scatter.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double scatterTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t *d_idx1, ssize_t streamArraySize,
                    STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[d_idx1[j]] = d_b[j] + scalar * d_c[j];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double sgCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
              ssize_t *d_idx1, ssize_t *d_idx2, ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[d_idx2[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double sgScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
               ssize_t *d_idx1, ssize_t *d_idx2, ssize_t streamArraySize,
               STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1, d_idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[d_idx2[j]] = scalar * d_c[d_idx1[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double sgAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
             ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
             ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[d_idx1[j]] = d_a[d_idx2[j]] + d_b[d_idx3[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double sgTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b, STREAM_TYPE *d_c,
               ssize_t *d_idx1, ssize_t *d_idx2, ssize_t *d_idx3,
               ssize_t streamArraySize, STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c, d_idx1, d_idx2, d_idx3)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[d_idx2[j]] = d_b[d_idx3[j]] + scalar * d_c[d_idx1[j]];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Copies data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double centralCopy(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                   STREAM_TYPE *d_c, ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[0] = d_a[0];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Scales data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double centralScale(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t streamArraySize, STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_b[0] = scalar * d_c[0];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Adds data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 **************************************************/

double centralAdd(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                  STREAM_TYPE *d_c, ssize_t streamArraySize) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_c[0] = d_a[0] + d_b[0];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/**************************************************
 * @brief Performs triad operation using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 **************************************************/

double centralTriad(STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                    STREAM_TYPE *d_c, ssize_t streamArraySize, STREAM_TYPE scalar) {
  double start = mySecond();
#pragma acc parallel loop \
    deviceptr(d_a, d_b, d_c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    d_a[0] = d_b[0] + scalar * d_c[0];
  double end = mySecond();
  double time = (end - start);
  return time;
}

/* EOF */
