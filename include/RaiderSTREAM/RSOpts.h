//
// _RSOPTS_H_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#ifndef _RSOPTS_H_
#define _RSOPTS_H_

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <getopt.h>

#include "RSBaseImpl.h"

#define RS_VERSION_MAJOR 0
#define RS_VERSION_MINOR 2

/**
 * @brief BenchType struct: defines an individual benchmark table entry
 *
 * This structure defines the attributes of a benchmark entry, including its name,
 * arguments, notes, kernel type, and whether it is enabled and has required arguments.
 */
typedef struct {
  const std::string Name;         ///< Benchmark name
  std::string Arg;                ///< Benchmark arguments
  std::string Notes;              ///< Benchmark notes
  RSBaseImpl::RSKernelType KType; ///< Benchmark kernel type
  bool Enabled;                   ///< Flag indicating if the benchmark is enabled
  bool ReqArq;                    ///< Flag indicating if the benchmark has required arguments
} BenchType;

extern BenchType BenchTypeTable[];

/**
 * @brief RSOpts class: manages command-line options and benchmark settings
 *
 * This class provides functionality for parsing command-line options, managing benchmark
 * settings, and storing benchmark-related arrays.
 */
class RSOpts {
private:
  RSBaseImpl::RSKernelType kernelType = RSBaseImpl::RS_ALL;
  bool isHelp = false;                    ///< Flag to determine if the help option has been selected
  bool isList = false;                    ///< Flag to list the benchmarks
  std::string kernelName;                 ///< Kernel name
  ssize_t streamArraySize = 1000000;      ///< STREAM array size
  int numPEs = 1;                             ///< Number of PEs
  int lArgc;                              ///< Main argc
  char **lArgv;                           ///< Main argv

  void printHelp();

  void printVersion();

  void printBench();

  bool enableBench(std::string BenchName);

public:
  RSOpts();
  ~RSOpts();

  bool parseOpts(int argc, char **argv);

  bool enableBenchmark(std::string benchName);

  /**
   * @brief Array of memory transfer times (bytes).
   *
   * This array stores the memory transfer times for different benchmark kernels.
   */
  double BYTES[NUM_KERNELS] = {
		// Original Kernels
		static_cast<double>(2 * sizeof(double) * streamArraySize), // Copy
		static_cast<double>(2 * sizeof(double) * streamArraySize), // Scale
		static_cast<double>(3 * sizeof(double) * streamArraySize), // Add
		static_cast<double>(3 * sizeof(double) * streamArraySize), // Triad
		// Gather Kernels
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t))) * streamArraySize), // GATHER Copy
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t))) * streamArraySize), // GATHER Scale
		static_cast<double>(((3 * sizeof(double)) + (2 * sizeof(ssize_t))) * streamArraySize), // GATHER Add
		static_cast<double>(((3 * sizeof(double)) + (2 * sizeof(ssize_t))) * streamArraySize), // GATHER Triad
		// Scatter Kernels
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Copy
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Scale
		static_cast<double>(((3 * sizeof(double)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Add
		static_cast<double>(((3 * sizeof(double)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Triad
		// Scatter-Gather Kernels
		static_cast<double>(((2 * sizeof(double)) + (2 * sizeof(ssize_t))) * streamArraySize), // SG Copy
		static_cast<double>(((2 * sizeof(double)) + (2 * sizeof(ssize_t))) * streamArraySize), // SG Scale
		static_cast<double>(((3 * sizeof(double)) + (3 * sizeof(ssize_t))) * streamArraySize), // SG Add
		static_cast<double>(((3 * sizeof(double)) + (3 * sizeof(ssize_t))) * streamArraySize), // SG Triad
		// Central Kernels
		static_cast<double>(2 * sizeof(double) * streamArraySize), // Central Copy
		static_cast<double>(2 * sizeof(double) * streamArraySize), // Central Scale
		static_cast<double>(3 * sizeof(double) * streamArraySize), // Central Add
		static_cast<double>(3 * sizeof(double) * streamArraySize), // Central Triad
  };

  double FLOATOPS[NUM_KERNELS] = {
		// Original Kernels
		(double)0.0,                       // Copy
		1.0 * streamArraySize,             // Scale
		1.0 * streamArraySize,             // Add
		2.0 * streamArraySize,             // Triad
		// Gather Kernels
		(double)0.0,                       // GATHER Copy
		1.0 * streamArraySize,             // GATHER Scale
		1.0 * streamArraySize,             // GATHER Add
		2.0 * streamArraySize,             // GATHER Triad
		// Scatter Kernels
		(double)0.0,                       // SCATTER Copy
		1.0 * streamArraySize,             // SCATTER Scale
		1.0 * streamArraySize,             // SCATTER Add
		2.0 * streamArraySize,             // SCATTER Triad
		// Scatter-Gather Kernels
		(double)0.0,                       // SG Copy
		1.0 * streamArraySize,             // SG Scale
		1.0 * streamArraySize,             // SG Add
		2.0 * streamArraySize,             // SG Triad
		// Central Kernels
		(double)0.0,                       // CENTRAL Copy
		1.0 * streamArraySize,             // CENTRAL Scale
		1.0 * streamArraySize,             // CENTRAL Add
		2.0 * streamArraySize,             // CENTRAL Triad
  };

	double MBPS[NUM_KERNELS];

	double FLOPS[NUM_KERNELS];

	double TIMES[NUM_KERNELS];

/****************************************************
 * 									 Getters 
****************************************************/
  int getArgc() { return lArgc; }

  char **getArgv() { return lArgv; }

  bool getIsHelp() { return isHelp; }

  bool getIsList() { return isList; }

  RSBaseImpl::RSKernelType getKernelType() const { return kernelType; }

  std::string getKernelName() { return kernelName; }

  ssize_t getStreamArraySize() { return streamArraySize; }

  int getNumPEs() { return numPEs; }

  // FIXME: not sure if we need the below two arrays
  double* getBytes() { return BYTES; }

  double* getFloatOps() { return FLOATOPS; }

/****************************************************
 * 									 Setters
****************************************************/
  void setStreamArraySize(ssize_t size) { streamArraySize = size; }

  void setNumPEs(int nprocs) { numPEs = nprocs; }

  void setKernelType(RSBaseImpl::RSKernelType type) { kernelType = type; }
};

#endif // _RSOPTS_H_
