//
// _RSBASEIMPL_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifndef _RSBASEIMPL_H_
#define _RSBASEIMPL_H_

#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <sys/types.h>

#ifndef NUM_KERNELS
#define NUM_KERNELS 20
#endif

#ifndef NUM_ARRAYS
#define NUM_ARRAYS 3
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#define M 20

/**
 * @brief RSBaseImpl class: Base class for RaiderSTREAM implementations
 *
 * This class serves as the base class for RaiderSTREAM benchmark implementations.
 * It includes various utility functions and defines constants for benchmarking.
 */
class RSBaseImpl {
public:
  typedef enum {
    RS_SEQ_COPY = 0,
    RS_SEQ_SCALE = 1,
    RS_SEQ_ADD = 2,
    RS_SEQ_TRIAD = 3,

    RS_GATHER_COPY = 4,
    RS_GATHER_SCALE = 5,
    RS_GATHER_ADD = 6,
    RS_GATHER_TRIAD = 7,

    RS_SCATTER_COPY = 8,
    RS_SCATTER_SCALE = 9,
    RS_SCATTER_ADD = 10,
    RS_SCATTER_TRIAD = 11,

    RS_SG_COPY = 12,
    RS_SG_SCALE = 13,
    RS_SG_ADD = 14,
    RS_SG_TRIAD = 15,

    RS_CENTRAL_COPY = 16,
    RS_CENTRAL_SCALE = 17,
    RS_CENTRAL_ADD = 18,
    RS_CENTRAL_TRIAD = 19,
    RS_ALL = 20,
    RS_NB = 21
  } RSKernelType;

  // Default constructor
  RSBaseImpl(
    std::string implName,
    RSBaseImpl::RSKernelType kType)
    : Impl(implName), KType(kType
  ) {}

  // Default virtual destructor
  virtual ~RSBaseImpl() {}

  /**
   * @brief Get the implementation name.
   *
   * This method retrieves the name of the implementation.
   *
   * @return The implementation name.
   */
  std::string getImplName() { return Impl; }

  /**
   * @brief Allocate data for benchmarking.
   *
   * This pure virtual method allocates memory for benchmarking data and initializes necessary variables.
   *
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @param idx1 Pointer to index array 1.
   * @param idx2 Pointer to index array 2.
   * @param idx3 Pointer to index array 3.
   * @param avgTime Pointer to store the average execution time.
   * @param maxTime Pointer to store the maximum execution time.
   * @param minTime Pointer to store the minimum execution time.
   * @return True if data allocation is successful, false otherwise.
   */
  virtual bool allocateData(
    double *a, double *b, double *c,
    ssize_t *idx1, ssize_t *idx2, ssize_t *idx3,
    double *mbps, double *flops, double *times
    // double *avgTime, double *maxTime, double *minTime, double *times
  ) = 0;

  /**
   * @brief Free allocated data.
   *
   * This pure virtual method frees memory allocated for benchmarking data.
   */
  virtual bool freeData() = 0;

  /**
   * @brief Execute the benchmark.
   *
   * This pure virtual method performs the benchmark computation.
   *
   * @return True if the benchmark execution is successful, false otherwise.
   */
  virtual bool execute(double *times, double *mbps, double *flops) = 0;

  /**
   * @brief Check for errors in the benchmark results.
   *
   * This pure virtual method checks for errors in the benchmark results.
   *
   * @param label Label or description for the check.
   * @param array Pointer to the array to check.
   * @param avgErr Average error threshold.
   * @param expVal Expected value.
   * @param epsilon Epsilon value for comparison.
   * @param errors Pointer to store the number of errors found.
   * @param streamArraySize Size of the STREAM array.
   * @return True if no errors are found, false otherwise.
   */
  virtual bool checkErrors(
    const char *label,
    double *array,
    double avgErr,
    double expVal,
    double epsilon,
    int *errors,
    ssize_t streamArraySize) = 0;

  /**
   * @brief Check for errors in central benchmark results.
   *
   * This pure virtual method checks for errors in the central benchmark results.
   *
   * @param label Label or description for the check.
   * @param array Pointer to the array to check.
   * @param avgErr Average error threshold.
   * @param expVal Expected value.
   * @param epsilon Epsilon value for comparison.
   * @param errors Pointer to store the number of errors found.
   * @param streamArraySize Size of the STREAM array.
   * @return True if no errors are found, false otherwise.
   */
  virtual bool centralCheckErrors(
    const char *label,
    double *array,
    double avgErr,
    double expVal,
    double epsilon,
    int *errors,
    ssize_t streamArraySize) = 0;

  /**
   * @brief Compute standard errors for the benchmark.
   *
   * This pure virtual method computes standard errors for the benchmark.
   *
   * @param aj Index value for array A.
   * @param bj Index value for array B.
   * @param cj Index value for array C.
   * @param streamArraySize Size of the STREAM array.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @param aSumErr Pointer to store the sum of errors for array A.
   * @param bSumErr Pointer to store the sum of errors for array B.
   * @param cSumErr Pointer to store the sum of errors for array C.
   * @param aAvgErr Pointer to store the average error for array A.
   * @param bAvgErr Pointer to store the average error for array B.
   * @param cAvgErr Pointer to store the average error for array C.
   * @return True if computation is successful, false otherwise.
   */
  virtual bool standardErrors(
    double aj, double bj, double cj,
    ssize_t streamArraySize,
    double *a, double *b, double *c,
    double *aSumErr, double *bSumErr, double *cSumErr,
    double *aAvgErr, double *bAvgErr, double *cAvgErr) = 0;

  /**
   * @brief Validate values in the benchmark arrays.
   *
   * This pure virtual method validates values in the benchmark arrays.
   *
   * @param aj Index value for array A.
   * @param bj Index value for array B.
   * @param cj Index value for array C.
   * @param streamArraySize Size of the STREAM array.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @return True if values are valid, false otherwise.
   */
  virtual bool validateValues(
    double aj, double bj, double cj,
    ssize_t streamArraySize,
    double *a, double *b, double *c) = 0;

  /**
   * @brief Compute central errors for the benchmark.
   *
   * This pure virtual method computes central errors for the benchmark.
   *
   * @param aj Index value for array A.
   * @param bj Index value for array B.
   * @param cj Index value for array C.
   * @param streamArraySize Size of the STREAM array.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @param aSumErr Pointer to store the sum of errors for array A.
   * @param bSumErr Pointer to store the sum of errors for array B.
   * @param cSumErr Pointer to store the sum of errors for array C.
   * @param aAvgErr Pointer to store the average error for array A.
   * @param bAvgErr Pointer to store the average error for array B.
   * @param cAvgErr Pointer to store the average error for array C.
   * @return True if computation is successful, false otherwise.
   */
  virtual bool centralErrors(
    double aj, double bj, double cj,
    ssize_t streamArraySize,
    double *a, double *b, double *c,
    double *aSumErr, double *bSumErr, double *cSumErr,
    double *aAvgErr, double *bAvgErr, double *cAvgErr) = 0;

  /**
   * @brief Perform sequential validation.
   *
   * This pure virtual method performs sequential validation for the benchmark.
   *
   * @param streamArraySize Size of the STREAM array.
   * @param scalar Scalar value.
   * @param isValidated Pointer to store the validation result.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @return True if validation is successful, false otherwise.
   */
  virtual bool seqValidation(
    ssize_t streamArraySize,
    double scalar,
    int *isValidated,
    double *a, double *b, double *c) = 0;

  /**
   * @brief Perform gather validation.
   *
   * This pure virtual method performs gather validation for the benchmark.
   *
   * @param streamArraySize Size of the STREAM array.
   * @param scalar Scalar value.
   * @param isValidated Pointer to store the validation result.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @return True if validation is successful, false otherwise.
   */
  virtual bool gatherValidation(
    ssize_t streamArraySize,
    double scalar,
    int *isValidated,
    double *a, double *b, double *c) = 0;

  /**
   * @brief Perform scatter validation.
   *
   * This pure virtual method performs scatter validation for the benchmark.
   *
   * @param streamArraySize Size of the STREAM array.
   * @param scalar Scalar value.
   * @param isValidated Pointer to store the validation result.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @return True if validation is successful, false otherwise.
   */
  virtual bool scatterValidation(
    ssize_t streamArraySize,
    double scalar,
    int *isValidated,
    double *a, double *b, double *c) = 0;

  /**
   * @brief Perform scatter-gather validation.
   *
   * This pure virtual method performs scatter-gather validation for the benchmark.
   *
   * @param streamArraySize Size of the STREAM array.
   * @param scalar Scalar value.
   * @param isValidated Pointer to store the validation result.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @return True if validation is successful, false otherwise.
   */
  virtual bool sgValidation(
    ssize_t streamArraySize,
    double scalar,
    int *isValidated,
    double *a, double *b, double *c) = 0;

  /**
   * @brief Perform central validation.
   *
   * This pure virtual method performs central validation for the benchmark.
   *
   * @param streamArraySize Size of the STREAM array.
   * @param scalar Scalar value.
   * @param isValidated Pointer to store the validation result.
   * @param a Pointer to array A.
   * @param b Pointer to array B.
   * @param c Pointer to array C.
   * @return True if validation is successful, false otherwise.
   */
  virtual bool centralValidation(
    ssize_t streamArraySize,
    double scalar,
    int *isValidated,
    double *a, double *b, double *c) = 0;

  /**
   * @brief Check the results of the STREAM benchmark.
   *
   * This pure virtual method checks the results of the STREAM benchmark.
   *
   * @param isValidated Pointer to store the validation result.
   * @return True if validation is successful, false otherwise.
   */
  virtual bool checkSTREAMResults(int *isValidated) = 0;

  /**
   * @brief Initialize a random index array.
   *
   * This method initializes a random index array.
   *
   * @param array Pointer to the index array.
   * @param nelems Number of elements in the array.
   */
  void initRandomIdxArray(ssize_t *array, ssize_t nelems) {
    int success;
    ssize_t i, idx;
    char *flags = (char *)malloc(sizeof(char) * nelems);
    for (i = 0; i < nelems; i++)
      flags[i] = 0;
    for (i = 0; i < nelems; i++) {
      success = 0;
      while (success == 0) {
        idx = ((ssize_t)rand()) % nelems;
        if (flags[idx] == 0) {
          array[i] = idx;
          flags[idx] = -1;
          success = 1;
        }
      }
    }
    free(flags);
  }

  /**
   * @brief Initialize an index array from a file.
   *
   * This method initializes an index array by reading values from a file.
   *
   * @param array Pointer to the index array to be initialized.
   * @param nelems Number of elements in the array.
   * @param filename Name of the file from which to read the values.
   */
  void initReadIdxArray(ssize_t *array, ssize_t nelems, char *filename) {
    FILE *file;
    file = fopen(filename, "r");
    if (!file) {
      perror(filename);
      exit(1);
    }
    for (ssize_t i = 0; i < nelems; i++)
      fscanf(file, "%zd", &array[i]);

    fclose(file);
  }

  /**
   * @brief Initialize a double array with a specified value.
   *
   * This method initializes a double array by setting all elements to the specified value.
   *
   * @param array Pointer to the double array to be initialized.
   * @param arrayElements Number of elements in the array.
   * @param value The value to assign to each element of the array.
   */
  void initStreamArray(double *array, ssize_t arrayElements, double value) {
    for (ssize_t i = 0; i < arrayElements; i++)
      array[i] = value;
  }

  /**
   * @brief Get the current system time in seconds.
   *
   * This method retrieves the current system time in seconds since the epoch.
   *
   * @return The current system time in seconds.
   */
  double mySecond() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
  }

  /**
   * @brief Check the minimum time resolution of the system's timer.
   *
   * This method checks the minimum time resolution of the system's timer by measuring time intervals.
   *
   * @return The minimum time resolution in microseconds.
   */
  int checkTick() {
    int i, minDelta, delta;
    double t1, t2, timesFound[M];
    for (i = 0; i < M; i++) {
      t1 = mySecond();
      while (((t2 = mySecond()) - t1) < 1.0E-6)
        ;
      timesFound[i] = t1 = t2;
    }
    minDelta = 1000000;
    for (i = 1; i < M; i++) {
      delta = (int)(1.0E6 * (timesFound[i] - timesFound[i - 1]));
      minDelta = MIN(minDelta, MAX(delta, 0));
    }
    return (minDelta);
  }

  /**
   * @brief Calculate the runtime in seconds.
   *
   * This method calculates the runtime in seconds given a start and end time.
   *
   * @param startTime The start time in seconds.
   * @param endTime The end time in seconds.
   * @return The runtime in seconds.
   */
  double calculateRunTime(double startTime, double endTime) {
    return (endTime - startTime);
  }

  /**
   * @brief Calculate the bandwidth in MB/s.
   *
   * This method calculates the bandwidth in megabytes per second (MB/s) based on the given parameters.
   * 
   * FIXME: this is wrong. Pass is the bytes array as an argument
   *
   * @param streamArraySize Number of elements per STREAM array.
   * @param runTime The runtime in seconds.
   * @return The throughput in MB/s.
   */
  double calculateMBPS(double streamArraySize, double runTime) {
    return (1.0E-06 * streamArraySize / runTime);
  }

  /**
   * @brief Calculate the FLOP/s.
   *
   * This method calculates the FLOP/s based on the given parameters.
   * 
   * FIXME: this is wrong. Pass is the floatOps array as an argument
   *
   * @param streamArraySize Number of elements per STREAM array.
   * @param runTime The runtime in seconds.
   * @return The FLOP/s.
   */
  double calculateFLOPS(double streamArraySize, double runTime) {
    return (2.0E-06 * streamArraySize / runTime);
  }

  /**
   * @brief Get the kernel type.
   *
   * This method retrieves the kernel type associated with the implementation.
   *
   * @return The kernel type.
   */
  RSBaseImpl::RSKernelType getKernelType() { return KType; }

private:
  std::string Impl;
  RSBaseImpl::RSKernelType KType;
};

#endif // _RSBASEIMPL_H_

// EOF
