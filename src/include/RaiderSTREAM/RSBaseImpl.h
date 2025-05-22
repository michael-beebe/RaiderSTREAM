/**
 * @file RSBaseImpl.h
 * @brief Base implementation class for RaiderSTREAM benchmark
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights
 * Reserved.
 * @author michael.beebe@ttu.edu
 *
 * See LICENSE in the top level directory for licensing details
 */

#ifndef _RSBASEIMPL_H_
#define _RSBASEIMPL_H_

#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>

#include <cstring>
#include <limits>
#include <vector>

/**
 * @brief Number of benchmark kernels to run
 */
#ifndef NUM_KERNELS
#define NUM_KERNELS 20
#endif

/**
 * @brief Number of arrays used in benchmark
 */
#ifndef NUM_ARRAYS
#define NUM_ARRAYS 3
#endif

/**
 * @brief Returns minimum of two values
 * @param x First value
 * @param y Second value
 * @return Minimum of x and y
 */
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

/**
 * @brief Returns maximum of two values
 * @param x First value
 * @param y Second value
 * @return Maximum of x and y
 */
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

/**
 * @brief Returns absolute value
 * @param a Input value
 * @return Absolute value of a
 */
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

// /**
//  * @brief Number of timing samples to collect
//  */
// #define M 20

/**
 * @brief RSBaseImpl class: Base class for RaiderSTREAM implementations
 *
 * This class serves as the base class for RaiderSTREAM benchmark
 * implementations. It includes various utility functions and defines constants
 * for benchmarking.
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
    RS_SEQ_COPY = 0,  /**< Sequential copy kernel */
    RS_SEQ_SCALE = 1, /**< Sequential scale kernel */
    RS_SEQ_ADD = 2,   /**< Sequential add kernel */
    RS_SEQ_TRIAD = 3, /**< Sequential triad kernel */

    RS_GATHER_COPY = 4,  /**< Gather copy kernel */
    RS_GATHER_SCALE = 5, /**< Gather scale kernel */
    RS_GATHER_ADD = 6,   /**< Gather add kernel */
    RS_GATHER_TRIAD = 7, /**< Gather triad kernel */

    RS_SCATTER_COPY = 8,   /**< Scatter copy kernel */
    RS_SCATTER_SCALE = 9,  /**< Scatter scale kernel */
    RS_SCATTER_ADD = 10,   /**< Scatter add kernel */
    RS_SCATTER_TRIAD = 11, /**< Scatter triad kernel */

    RS_SG_COPY = 12,  /**< Scatter-gather copy kernel */
    RS_SG_SCALE = 13, /**< Scatter-gather scale kernel */
    RS_SG_ADD = 14,   /**< Scatter-gather add kernel */
    RS_SG_TRIAD = 15, /**< Scatter-gather triad kernel */

    RS_CENTRAL_COPY = 16,  /**< Central copy kernel */
    RS_CENTRAL_SCALE = 17, /**< Central scale kernel */
    RS_CENTRAL_ADD = 18,   /**< Central add kernel */
    RS_CENTRAL_TRIAD = 19, /**< Central triad kernel */
    RS_ALL = 20,           /**< Run all kernels */
    RS_NB = 21             /**< Invalid kernel type */
  } RSKernelType;

  /**
   * @brief RSBaseImpl class: Constructor for RSBaseImpl
   *
   * This constructor initializes the RSBaseImpl object with the provided
   * implementation name and kernel type.
   *
   * @param implName The name of the implementation.
   * @param kType The kernel type for the implementation.
   */
  RSBaseImpl(const std::string &implName, RSKernelType kType)
      : Impl(implName), KType(kType) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~RSBaseImpl() {}

  /**
   * @brief getImplName()
   *
   * Returns the name of the current implementation.
   *
   * This function returns the name of the specific implementation that is being
   * used.
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
  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS,
                       double *BYTES, double *FLOATOPS) = 0;

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
      std::cerr << "Error: Array size too large to allocate flags array."
                << std::endl;
      return;
    }
    int success;
    ssize_t i, idx;
    std::vector<unsigned char> flags(
        nelems, 0); // Use std::vector to avoid allocation warnings
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
   * @brief Initializes a STREAM_TYPE array with a specific value
   *
   * This function initializes a STREAM_TYPE array with a specified value.
   *
   * @param array Pointer to the STREAM_TYPE array to be initialized
   * @param arrayElements Number of elements in the array
   * @param value Value to initialize the array with
   */
  void initStreamArray(STREAM_TYPE *array, ssize_t arrayElements,
                       STREAM_TYPE value) {
    for (ssize_t i = 0; i < arrayElements; i++)
      array[i] = value;
  }

  /**
   * @brief Produces a number repesenting the current time.
   *
   * @returns A number representing the time since some reference point.
   */
  double mySecond() {
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return ((double)tp.tv_sec + (double)tp.tv_nsec * 1.e-9);
  }

  /**
   * @brief Calculate the minimum difference in time.
   *
   * In other words, let x, y be two distinct results of mySecond.
   * |x - y| >= checkTick()
   *
   * @returns The minimum time difference detectable
   */
  double checkTick() {
    struct timespec tp;
    clock_getres(CLOCK_MONOTONIC, &tp);
    return ((double)tp.tv_sec + (double)tp.tv_nsec * 1.e-9);
  }

  /**
   * @brief Calculate the difference between two times.
   *
   * @param startTime The result of the first call to mySecond.
   * @param endTime The result of the second call to mySecond.
   * @return The difference between the two times in seconds.
   */
  double calculateRunTime(double startTime, double endTime) {
    return (endTime - startTime);
  }

  /**
   * @brief Calculate the effective MB/s given a bytes and a runtime.
   *
   * @param bytes The amount of bytes moved during the operation.
   * @param runTime Duration of the operation in seconds.
   * @return The effective memory bandwidth in MB/s.
   */
  double calculateMBPS(double bytes, double runTime) {
    return (bytes / (runTime * 1024.0 * 1024.0));
  }

  /**
   * @brief Calculate the effective FLOPS given a bytes and a runtime.
   *
   * @param floatOps The amount of floating point operations performed.
   * @param runTime Duration of the operation in seconds.
   * @return The effective floating point operations per second (FLOPS).
   */
  double calculateFLOPS(double floatOps, double runTime) {
    return (floatOps / runTime);
  }

  /**
   * @brief Return the kernel chosen to run.
   *
   * @return The kernel type that will be run.
   */
  RSBaseImpl::RSKernelType getKernelType() { return KType; }

private:
  std::string Impl;               /**< Name of the implementation */
  RSBaseImpl::RSKernelType KType; /**< Type of kernel to run */
};

#endif // _RSBASEIMPL_H_

// EOF
