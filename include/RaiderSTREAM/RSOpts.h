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

#include "RaiderSTREAM/RSBaseImpl.h"

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

/**
 * @brief RSOpts class: manages command-line options and benchmark settings
 *
 * This class provides functionality for parsing command-line options, managing benchmark
 * settings, and storing benchmark-related arrays.
 */
class RSOpts {
private:
  bool isHelp;                    ///< Flag to determine if the help option has been selected
  bool isList;                    ///< Flag to list the benchmarks
  std::string kernelName;         ///< Kernel name
  ssize_t streamArraySize;        ///< STREAM array size
  int numTimes;                   ///< Number of times to run the benchmark
  std::string streamType;         ///< STREAM datatype
  int numPEs;                     ///< Number of PEs
  int lArgc;                      ///< Main argc
  char **lArgv;                   ///< Main argv

  /**
   * @brief Print help information.
   *
   * This method prints information about available command-line options and usage.
   */
  void printHelp();

  /**
   * @brief Print the version information.
   *
   * This method prints the version information of the program.
   */
  void printVersion();

  /**
   * @brief Print benchmark information.
   *
   * This method prints information about the available benchmarks.
   */
  void printBench();

  /**
   * @brief Enable a benchmark.
   *
   * This method enables a benchmark by its name.
   *
   * @param BenchName The name of the benchmark to enable.
   * @return True if the benchmark is successfully enabled, false otherwise.
   */
  bool enableBench(std::string BenchName);

public:
  /**
   * @brief Default constructor for RSOpts.
   */
  RSOpts();

  /**
   * @brief Default destructor for RSOpts.
   */
  ~RSOpts();

  /**
   * @brief Parse command-line options.
   *
   * This method parses the command-line options and sets the corresponding class members.
   *
   * @param argc The number of command-line arguments.
   * @param argv The array of command-line argument strings.
   * @return True if parsing is successful, false otherwise.
   */
  bool parseOpts(int argc, char **argv);

  /**
   * @brief Enable a benchmark by name.
   *
   * This method enables a benchmark based on its name.
   *
   * @param benchName The name of the benchmark to enable.
   * @return True if the benchmark is successfully enabled, false otherwise.
   */
  bool enableBenchmark(std::string benchName);

  /**
   * @brief Array of memory transfer times (bytes).
   *
   * This array stores the memory transfer times for different benchmark kernels.
   */
  double bytes[NUM_KERNELS] = {
		// Original Kernels
		static_cast<double>(2 * sizeof(streamType) * streamArraySize), // Copy
		static_cast<double>(2 * sizeof(streamType) * streamArraySize), // Scale
		static_cast<double>(3 * sizeof(streamType) * streamArraySize), // Add
		static_cast<double>(3 * sizeof(streamType) * streamArraySize), // Triad
		// Gather Kernels
		static_cast<double>(((2 * sizeof(streamType)) + (1 * sizeof(ssize_t))) * streamArraySize), // GATHER Copy
		static_cast<double>(((2 * sizeof(streamType)) + (1 * sizeof(ssize_t))) * streamArraySize), // GATHER Scale
		static_cast<double>(((3 * sizeof(streamType)) + (2 * sizeof(ssize_t))) * streamArraySize), // GATHER Add
		static_cast<double>(((3 * sizeof(streamType)) + (2 * sizeof(ssize_t))) * streamArraySize), // GATHER Triad
		// Scatter Kernels
		static_cast<double>(((2 * sizeof(streamType)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Copy
		static_cast<double>(((2 * sizeof(streamType)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Scale
		static_cast<double>(((3 * sizeof(streamType)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Add
		static_cast<double>(((3 * sizeof(streamType)) + (1 * sizeof(ssize_t))) * streamArraySize), // SCATTER Triad
		// Scatter-Gather Kernels
		static_cast<double>(((2 * sizeof(streamType)) + (2 * sizeof(ssize_t))) * streamArraySize), // SG Copy
		static_cast<double>(((2 * sizeof(streamType)) + (2 * sizeof(ssize_t))) * streamArraySize), // SG Scale
		static_cast<double>(((3 * sizeof(streamType)) + (3 * sizeof(ssize_t))) * streamArraySize), // SG Add
		static_cast<double>(((3 * sizeof(streamType)) + (3 * sizeof(ssize_t))) * streamArraySize), // SG Triad
		// Central Kernels
		static_cast<double>(2 * sizeof(streamType) * streamArraySize), // Central Copy
		static_cast<double>(2 * sizeof(streamType) * streamArraySize), // Central Scale
		static_cast<double>(3 * sizeof(streamType) * streamArraySize), // Central Add
		static_cast<double>(3 * sizeof(streamType) * streamArraySize), // Central Triad
  };

  /**
   * @brief Array of FLOP (floating-point operations) counts.
   *
   * This array stores the FLOP counts for different benchmark kernels.
   */
  double floatOps[NUM_KERNELS] = {
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

	/**
	 * @brief Array that stores MB/s.
	 *
	 * This array stores the megabytes per second for each benchmark kernel.
	 */
	double MBPS[NUM_KERNELS];

	/**
	 * @brief Array that stores FLOP/s.
	 *
	 * This array stores the FLOP/s for each kernel.
	 */
	double FLOPS[NUM_KERNELS];

	/**
	 * @brief Array that stores the average times.
	 *
	 * This array stores the average times for each kernel.
	 */
	double avgTimes[NUM_KERNELS];

	/**
	 * @brief Array that stores the maximum times.
	 *
	 * This array stores the maximum times for each kernel.
	 */
	double maxTimes[NUM_KERNELS];

	/**
	 * @brief Array that stores the minimum times.
	 *
	 * This array stores the minimum times for each kernel.
	 */
	double minTimes[NUM_KERNELS];

	/**
	 * @brief Array that stores the standard times.
	 *
	 * This array stores the standard times for each kernel.
	 */
	double times[NUM_KERNELS];

/****************************************************
 * 									 Getters 
****************************************************/
  /**
   * @brief Check if the help option is selected.
   *
   * @return True if the help option is selected, false otherwise.
   */
  bool getIsHelp() { return isHelp; }

  /**
   * @brief Check if the list option is selected.
   *
   * @return True if the list option is selected, false otherwise.
   */
  bool getIsList() { return isList; }

  /**
   * @brief Get the kernel name.
   *
   * @return The kernel name.
   */
  std::string getKernelName() { return kernelName; }

  /**
   * @brief Get the STREAM array size.
   *
   * @return The STREAM array size as a ssize_t.
   */
  ssize_t getStreamArraySize() { return streamArraySize; }

  /**
   * @brief Get the number of times to run the benchmark.
   *
   * @return The number of times to run the benchmark.
   */
  int getNumTimes() { return numTimes; }

  /**
   * @brief Get the number of ranks, PEs, etc.
   *
   * @return The number of PEs.
   */
  int getNumPEs() { return numPEs; }

  /**
   * @brief Retrieve the argc value.
   *
   * @return The argc value.
   */
  int getArgc() { return lArgc; }

  /**
   * @brief Retrieve the argv value.
   *
   * @return The argv value.
  */
  char **getArgv() { return lArgv; }

/****************************************************
 * 									 Setters
****************************************************/
  /**
   * @brief Set the STREAM array size.
   *
   * @param size The new STREAM array size to set.
   */
  void setStreamArraySize(ssize_t size) { streamArraySize = size; }

  /**
   * @brief Set the number of times to run the benchmark.
   *
   * @param ntimes The number of times to set.
   */
  void setNumTimes(int ntimes) { numTimes = ntimes; }

  /**
   * @brief Set the number of ranks, PEs, etc.
   *
   * @param nprocs The number of PEs to set.
   */
  void setNumPEs(int nprocs) { numPEs = nprocs; }

  /**
   * @brief Set the STREAM datatype.
   *
   * @param type The STREAM datatype to set.
   */
  void setStreamType(std::string type) { streamType = type; }
};

#endif // _RSOPTS_H_
