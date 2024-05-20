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
  RSBaseImpl(const std::string& implName, RSKernelType kType)
    : Impl(implName), KType(kType) {}

  // Default virtual destructor
  virtual ~RSBaseImpl() {}

  std::string getImplName() { return Impl; }


  virtual bool allocateData() = 0;
  // virtual bool allocateData(
  //   double*& a, double*& b, double*& c,
  //   ssize_t*& idx1, ssize_t*& idx2, ssize_t*& idx3,
  //   ssize_t streamArraySize
  // ) = 0;

  virtual bool freeData() = 0;
  // virtual bool freeData(
  //   double *a, double *b, double *c,
  //   ssize_t *idx1, ssize_t *idx2, ssize_t *idx3
  // ) = 0;

  virtual bool execute(double *TIMES, double *MBPS, double *FLOPS, double *BYTES, double *FLOATOPS) = 0;

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

  void initStreamArray(double *array, ssize_t arrayElements, double value) {
    for (ssize_t i = 0; i < arrayElements; i++)
      array[i] = value;
  }

  double mySecond() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
  }

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

  double calculateRunTime(double startTime, double endTime) {
    return (endTime - startTime);
  }

  double calculateMBPS(double bytes, double runTime) {
    return (bytes / (runTime * 1024.0 * 1024.0));
  }

  double calculateFLOPS(double floatOps, double runTime) {
    return (floatOps / runTime);
  }

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
