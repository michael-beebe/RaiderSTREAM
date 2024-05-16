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
#include <iomanip>
#include <string>

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

	// double MBPS[NUM_KERNELS];
	// double FLOPS[NUM_KERNELS];
	// double TIMES[NUM_KERNELS];

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
   * @return True if data allocation is successful, false otherwise.
   */
  virtual bool allocateData(
    double *a, double *b, double *c,
    ssize_t *idx1, ssize_t *idx2, ssize_t *idx3
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
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) = 0;

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
   *
   * @param streamArraySize Number of elements per STREAM array.
   * @param runTime The runtime in seconds.
   * @return The throughput in MB/s.
   */
  double calculateMBPS(double bytes, double runTime) {
    return (bytes / (runTime * 1024.0 * 1024.0));
  }

  /**
   * @brief Calculate the FLOP/s.
   *
   * This method calculates the FLOP/s based on the given parameters.
   * 
   *
   * @param streamArraySize Number of elements per STREAM array.
   * @param runTime The runtime in seconds.
   * @return The FLOP/s.
   */
  double calculateFLOPS(double floatOps, double runTime) {
    return (floatOps / runTime);
  }

  /**
   * @brief Get the kernel type.
   *
   * This method retrieves the kernel type associated with the implementation.
   *
   * @return The kernel type.
   */
  RSBaseImpl::RSKernelType getKernelType() { return KType; }

  void printResults(const std::string& kernelName, double totalRuntime, double mbps, double flops) {
    std::cout << std::left << std::setw(20) << kernelName;
    std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(6) << totalRuntime;
    std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(1) << mbps;
    std::cout << std::right << std::setw(20) << std::fixed << std::setprecision(1) << flops;
    std::cout << std::endl;
  }

  void printTable(const std::string& kernelName, double totalRuntime, const double* mbps, const double* flops) {
    std::cout << std::setfill('-') << std::setw(80) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << std::left << std::setw(20) << "Kernel";
    std::cout << std::right << std::setw(20) << "Total Runtime (s)";
    std::cout << std::right << std::setw(20) << "MB/s";
    std::cout << std::right << std::setw(20) << "FLOP/s";
    std::cout << std::endl;
    std::cout << std::setfill('-') << std::setw(80) << "-" << std::endl;
    std::cout << std::setfill(' ');

    printResults(kernelName, totalRuntime, mbps[RSBaseImpl::RS_SEQ_COPY], flops[RSBaseImpl::RS_SEQ_COPY]);

    std::cout << std::setfill('-') << std::setw(80) << "-" << std::endl;
  }

private:
  std::string Impl;
  RSBaseImpl::RSKernelType KType;
};

#endif // _RSBASEIMPL_H_

// EOF
