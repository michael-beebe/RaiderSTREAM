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
#define DO_PRAGMA(x) _Pragma(#x)
#endif

#define LOOP_PRAGMA(...) DO_PRAGMA(omp target teams distribute parallel for simd is_device_ptr(__VA_ARGS__) num_teams(16) num_threads(16))

/**************************************************
 * @brief Copies data from one stream to another.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void seqCopy(int nteams, int threads, double *a, double *b, double *c,
             ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j];
}

/**************************************************
 * @brief Scales data in a stream.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void seqScale(int nteams, int threads, double *a, double *b, double *c,
              ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[j];
}

/**************************************************
 * @brief Adds data from two streams.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void seqAdd(int nteams, int threads, double *a, double *b, double *c,
            ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation on stream data.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void seqTriad(int nteams, int threads, double *a, double *b, double *c,
              ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void gatherCopy(int nteams, int threads, double *a, double *b, double *c,
                ssize_t *idx1, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]];
}

/**************************************************
 * @brief Scales data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void gatherScale(int nteams, int threads, double *a, double *b, double *c,
                 ssize_t *idx1, ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[j] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void gatherAdd(int nteams, int threads, double *a, double *b, double *c,
               ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[j] = a[idx1[j]] + b[idx2[j]];
}

/**************************************************
 * @brief Performs triad operation using gather.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void gatherTriad(int nteams, int threads, double *a, double *b, double *c,
                 ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize,
                 double scalar) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[j] = b[idx1[j]] + scalar * c[idx2[j]];
}

/**************************************************
 * @brief Copies data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void scatterCopy(int nteams, int threads, double *a, double *b, double *c,
                 ssize_t *idx1, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j];
}

/**************************************************
 * @brief Scales data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void scatterScale(int nteams, int threads, double *a, double *b, double *c,
                  ssize_t *idx1, ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx1[j]] = scalar * c[j];
}

/**************************************************
 * @brief Adds data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void scatterAdd(int nteams, int threads, double *a, double *b, double *c,
                ssize_t *idx1, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[j] + b[j];
}

/**************************************************
 * @brief Performs triad operation using scatter.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void scatterTriad(int nteams, int threads, double *a, double *b, double *c,
                  ssize_t *idx1, ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c, idx1)
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx1[j]] = b[j] + scalar * c[j];
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void sgCopy(int nteams, int threads, double *a, double *b, double *c,
            ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]];
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void sgScale(int nteams, int threads, double *a, double *b, double *c,
             ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize,
             double scalar) {
  LOOP_PRAGMA(a, b, c, idx1, idx2)
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[idx2[j]] = scalar * c[idx1[j]];
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void sgAdd(int nteams, int threads, double *a, double *b, double *c,
           ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
           ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c, idx1, idx2, idx3)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void sgTriad(int nteams, int threads, double *a, double *b, double *c,
             ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
             ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c, idx1, idx2, idx3)
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
}

/**************************************************
 * @brief Copies data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void centralCopy(int nteams, int threads, double *a, double *b, double *c,
                 ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[0] = a[0];
}

/**************************************************
 * @brief Scales data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void centralScale(int nteams, int threads, double *a, double *b, double *c,
                  ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    b[0] = scalar * c[0];
}

/**************************************************
 * @brief Adds data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
void centralAdd(int nteams, int threads, double *a, double *b, double *c,
                ssize_t streamArraySize) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    c[0] = a[0] + b[0];
}

/**************************************************
 * @brief Performs triad operation using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
void centralTriad(int nteams, int threads, double *a, double *b, double *c,
                  ssize_t streamArraySize, double scalar) {
  LOOP_PRAGMA(a, b, c)
  for (ssize_t j = 0; j < streamArraySize; j++)
    a[0] = b[0] + scalar * c[0];
}

/* EOF */
