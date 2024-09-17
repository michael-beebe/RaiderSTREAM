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

#include <cstring>
#include <vector>
#include <limits>


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
  /**
   * @brief RSKernelType; enumeration of all kernels
   *
   * Note RS_ALL, which runs every kernel if passed to RSBaseImpl::execute.
   *
   * RS_ALL and RS_NB are invalid as an index into benchmark arrays.
   */
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

  /**
   * @brief RSBaseImpl class: Constructor for RSBaseImpl
   *
   * This constructor initializes the RSBaseImpl object with the provided implementation name and kernel type.
   *
   * @param implName The name of the implementation.
   * @param kType The kernel type for the implementation.
   */
  RSBaseImpl(const std::string& implName, RSKernelType kType)
    : Impl(implName), KType(kType) {}

  virtual ~RSBaseImpl() {}


  /**
   * @brief getImplName()
   *
   * Returns the name of the current implementation.
   *
   * This function returns the name of the specific implementation that is being used.
   *
   * @return The name of the current implementation.
   */
  std::string getImplName() { return Impl; }

  /**
   * @brief Allocate data for kernels.
   *
   * Depending on the implementation, this could
   * range from allocating data on an accelerator
   * or a group of machines or just on the local
   * host.
   *
   * @return True if successful, false otherwise.
   **/
  virtual bool allocateData() = 0;

  /**
   * @brief Free data for kernels.
   *
   * Depending on the implementation, this could
   * range from freeing data on an accelerator
   * or a group of machines or just on the local
   * host.
   *
   * @return True if successful, false otherwise.
   **/
  virtual bool freeData() = 0;

  /**
   * @brief Executes the specified kernel.
   *
   * @param TIMES Array to store the execution times
   *              for each kernel.
   * @param MBPS Array to store the memory bandwidths
   *             for each kernel.
   * @param FLOPS Array to store the floating-point
   *              operation counts for each kernel.
   * @param BYTES Array to store the byte sizes for
   *              each kernel.
   * @param FLOATOPS Array to store the floating-point
   *                 operation sizes for each kernel.
   *
   * @return True if the execution was successful,
   *         false otherwise.
   **/
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) = 0;

  /**
   * @brief Initializes an array with random indices.
   *
   * This function initializes the provided array with unique random indices.
   * It uses a rejection sampling approach to ensure that no index is repeated.
   *
   * All indices are within the range [0, nelems).
   *
   * @param array Pointer to the array to be initialized.
   * @param nelems Number of elements in the array.
   */
  void initRandomIdxArray(ssize_t *array, ssize_t nelems) {
    if (nelems > std::numeric_limits<ssize_t>::max() / sizeof(unsigned char)) {
      std::cerr << "Error: Array size too large to allocate flags array." << std::endl;
      return;
    }
    int success;
    ssize_t i, idx;
    std::vector<unsigned char> flags(nelems, 0); // Use std::vector to avoid allocation warnings
    for (i = 0; i < nelems; i++) {
      success = 0;
      while (success == 0) {
        idx = static_cast<ssize_t>(rand()) % nelems;
        if (flags[idx] == 0) {
          array[i] = idx;
          flags[idx] = 1;
          success = 1;
        }
      }
    }
  }

  /**
   * @brief Reads ARRAYGEN output indices into an array.
   *
   * No checks are made as to if the indices are in bounds.
   *
   * @param array Pointer to the array to be initialized.
   * @param nelems Number of elements in the array.
   * @param filename Name of the arraygen output file.
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
   * @brief Initializes a double array with a specific value
   *
   * This function initializes a double array with a specified value.
   *
   * @param array Pointer to the double array to be initialized
   * @param arrayElements Number of elements in the array
   * @param value Value to initialize the array with
   */
  void initStreamArray(STREAM_TYPE *array, ssize_t arrayElements, double value) {
    for (ssize_t i = 0; i < arrayElements; i++)
      array[i] = value;
  }


  /**
   * @brief Produces a number repesenting the current time.
   *
   * While fairly stable and reliable, the reference point (0 seconds)
   * is only stable within the same program execution.
   *
   * @returns A number representing the time since some reference point.
   */
  // Should this be remade to use the monotonic clock instead? (clock_get[res/time])
  // It would avoid the shenanigans occuring in checkTick below.
  double mySecond() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
  }

  /**
   * @brief Calculate the minimum difference in time.
   *
   * In other words: calculate the minimum x such that
   * y = mySecond(), for(i = 0; i < x; i++) ;, mySecond() - y > 0
   *
   * @returns The minimum
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
   * @brief Calculate the difference between two times.
   *
   * @param startTime The result of the first call to mySecond.
   * @param endTime The result of the second call to mySecond.
   * @return The difference between the two.
   */
  double calculateRunTime(double startTime, double endTime) {
    return (endTime - startTime);
  }

  /**
   * @brief Calculate the effective MB/s given a bytes and a runtime.
   *
   * @param bytes The amount of bytes moved during the operation.
   * @param runTime Duration of the operation.
   * @return The effective MBPS of the operation.
   */
  double calculateMBPS(double bytes, double runTime) {
    return (bytes / (runTime * 1024.0 * 1024.0));
  }

  /**
   * @brief Calculate the effective FLOPS given a bytes and a runtime.
   *
   * @param floatOps The amount of floating point operations performed.
   * @param runTime Duration of the operation.
   * @return The effective FLOPS of the operation.
   */
  double calculateFLOPS(double floatOps, double runTime) {
    return (floatOps / runTime);
  }

  /**
   * @brief Return the kernel chosen to run.
   *
   * @return The kernel that will be run.
   */
  RSBaseImpl::RSKernelType getKernelType() { return KType; }

private:
  std::string Impl;
  RSBaseImpl::RSKernelType KType;
};

#endif // _RSBASEIMPL_H_

// EOF
