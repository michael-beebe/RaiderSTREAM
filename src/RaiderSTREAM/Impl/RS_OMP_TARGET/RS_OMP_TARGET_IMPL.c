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

#define WITH_TARGET(...)                                                       \
  DO_PRAGMA(omp target map(tofrom : __VA_ARGS__) map(tofrom : time))

#define LOOP_PRAGMA                                                            \
  DO_PRAGMA(omp teams distribute parallel for simd num_teams(nteams) \
                                                   thread_limit(threads))

#define BENCH_PREAMBLE double start = omp_get_wtime()
#define BENCH_POSTAMBLE time = omp_get_wtime() - start

/**************************************************
 * @brief Copies data from one stream to another.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double seqCopy(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
               ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Scales data in a stream.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double seqScale(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[j] = scalar * c[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Adds data from two streams.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double seqAdd(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[j] + b[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Performs triad operation on stream data.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double seqTriad(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[j] = b[j] + scalar * c[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Copies data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double gatherCopy(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t *idx1, ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[idx1[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Scales data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double gatherScale(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                   ssize_t *idx1, ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize], streamArraySize,
              scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[j] = scalar * c[idx1[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Adds data using gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double gatherAdd(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                 ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              idx2 [0:streamArraySize], streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[j] = a[idx1[j]] + b[idx2[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Performs triad operation using gather.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double gatherTriad(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                   ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize,
                   STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              idx2 [0:streamArraySize], streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[j] = b[idx1[j]] + scalar * c[idx2[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Copies data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double scatterCopy(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                   ssize_t *idx1, ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Scales data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double scatterScale(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                    ssize_t *idx1, ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize], streamArraySize,
              scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[idx1[j]] = scalar * c[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Adds data using scatter operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double scatterAdd(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t *idx1, ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[j] + b[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Performs triad operation using scatter.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double scatterTriad(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                    ssize_t *idx1, ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize], streamArraySize,
              scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[idx1[j]] = b[j] + scalar * c[j];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double sgCopy(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
              ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              idx2 [0:streamArraySize], streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[idx2[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double sgScale(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
               ssize_t *idx1, ssize_t *idx2, ssize_t streamArraySize,
               STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              idx2 [0:streamArraySize], streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[idx2[j]] = scalar * c[idx1[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double sgAdd(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
             ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
             ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              idx2 [0:streamArraySize], idx3 [0:streamArraySize],
              streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[idx1[j]] = a[idx2[j]] + b[idx3[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double sgTriad(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
               ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
               ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], idx1 [0:streamArraySize],
              idx2 [0:streamArraySize], idx3 [0:streamArraySize],
              streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[idx2[j]] = b[idx3[j]] + scalar * c[idx1[j]];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Copies data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double centralCopy(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                   ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[0] = a[0];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Scales data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double centralScale(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                    ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      b[0] = scalar * c[0];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Adds data using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @return Internally measured benchmark runtime.
 **************************************************/
double centralAdd(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                  ssize_t streamArraySize) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      c[0] = a[0] + b[0];
    BENCH_POSTAMBLE;
  }
  return time;
}

/**************************************************
 * @brief Performs triad operation using a central location.
 *
 * @param streamArraySize Size of the stream array.
 * @param scalar Scalar value for operations.
 * @return Internally measured benchmark runtime.
 **************************************************/
double centralTriad(int nteams, int threads, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                    ssize_t streamArraySize, STREAM_TYPE scalar) {
  double time = 0;
  WITH_TARGET(a [0:streamArraySize], b [0:streamArraySize],
              c [0:streamArraySize], streamArraySize, scalar) {
    BENCH_PREAMBLE;
    LOOP_PRAGMA
    for (ssize_t j = 0; j < streamArraySize; j++)
      a[0] = b[0] + scalar * c[0];
    BENCH_POSTAMBLE;
  }
  return time;
}

/* EOF */
