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
#include <string>
#include <iomanip>


#include "RSBaseImpl.h"

#define RS_VERSION_MAJOR 1
#define RS_VERSION_MINOR 0

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
  bool isHelp = false;
  bool isList = false;
  std::string kernelName;
  ssize_t streamArraySize = 1000000;
  int numPEs = 1;
	int lArgc;
  char **lArgv;
	#if _ENABLE_CUDA_ || _ENABLE_MPI_CUDA_
		int threadBlocks;
		int threadsPerBlock;
	#endif

  /**
   * @brief Output the help dialog.
   */
  void printHelp();

  /**
   * @brief Output the version of RaiderSTREAM.
   */
  void printVersion();

  /**
   * @brief Output the kernel to-be-run.
   */
  void printBench();

  /**
   * @brief Set a kernel to be run.
   * @param BenchName The name of the kernel to be enabled.
   * @returns True if enabling the kernel was successful, false otherwise.
   */
  bool enableBench(std::string BenchName);

public:
  RSOpts();
  ~RSOpts();

  /**
   * @brief Parse options from the command line.
   * @param argc The argc from the runtime.
   * @param argv The argv from the runtime.
   * @returns True if successful, false otherwise.
   */
  bool parseOpts(int argc, char **argv);

  /**
   * @brief Set a benchmark to be run this execution.
   * @param benchName The name of the kernel to be enabled.
   * @returns True if enabling the kernel was successful, false otherwise.
   */
  bool enableBenchmark(std::string benchName);

  /**
   * @brief Print out the help dialog.
   */
	void printOpts();

  /**
   * @brief Print out the ASCII art logo.
   */
	void printLogo();

/****************************************************
 * 									 Getters 
****************************************************/
  /**
   * @brief Get the stored arg count.
   * @returns The arg count.
   */
  int getArgc() { return lArgc; }
  /**
   * @brief Get the stored arg list.
   * @returns The arg list.
   */
  char **getArgv() { return lArgv; }
  /**
   * @brief Query if the invocation asks for the help dialog.
   * @returns True if the invocation asks for the help dialog, false otherwise.
   */
  bool getIsHelp() { return isHelp; }
  /**
   * @brief Query if the invocation asks for the kernel list.
   * @returns True if the invocation asks for the kernel list, false otherwise.
   */
  bool getIsList() { return isList; }

  /**
   * @brief Which kernel is being run this invocation.
   * @returns The kernel type the cli args have selected.
   */
  RSBaseImpl::RSKernelType getKernelType() const { return getKernelTypeFromName(kernelName); }

  /**
   * @brief The name of the kernel being run this invocation.
   * @returns The name of the kernel being run this invocation.
   */
  std::string getKernelName() const { return kernelName; }

  /**
   * @brief Parse a kernel name to a kernel type.
   * @param kernelName The name of the kernel.
   * @returns The kernel type if recognized, or RS_NB otherwise.
   */
	RSBaseImpl::RSKernelType getKernelTypeFromName(const std::string& kernelName) const;

  /**
   * @brief The size of the data arrays for this invocation.
   * @returns The size of the data arrays for this invocation.
   */
  ssize_t getStreamArraySize() const { return streamArraySize; }

  /**
   * @brief Gets the number of PEs specified in the command line.
   * @returns The number of PEs specified in the command line.
   */
  int getNumPEs() const { return numPEs; }

	#if _ENABLE_CUDA_ || _ENABLE_MPI_CUDA_ || _ENABLE_OACC_
    /**
     * @brief Gets the number of work groups.
     *
     * Blocks in CUDA, Groups in OpenMP, Gangs in OpenACC.
     *
     * @returns The number of working groups.
     */
		int getThreadBlocks() const { return threadBlocks; }
    /**
     * @brief Gets the number of workers per work group.
     *
     * Warps in CUDA, Workers in OpenMP and OpenACC.
     *
     * @returns The number of workers per working group.
     */
		int getThreadsPerBlocks() const { return threadsPerBlock; }
	#endif

/****************************************************
 * 									 Setters
****************************************************/
  /**
   * @brief Sets the size of the data array.
   * @param size The new size of the array in elements.
   */
  void setStreamArraySize(ssize_t size) { streamArraySize = size; }

  /**
   * @brief Sets the amount of PEs.
   * @param nprocs The new amount of PEs.
   */
  void setNumPEs(int nprocs) { numPEs = nprocs; }

  /**
   * @brief Sets the kernel to be run.
   * @param type The new type to be run.
   */
  void setKernelType(RSBaseImpl::RSKernelType type) { kernelType = type; }

  /**
   * @brief Sets the name of the kernel to be run.
   * @param name The new name of the kernel.
   */
	void setKernelName(std::string name) { kernelName = name; }

	#if _ENABLE_CUDA_ || _ENABLE_MPI_CUDA_ || _ENABLE_OACC_
    /**
     * @brief Sets the number of work groups.
     *
     * Blocks in CUDA, Groups in OpenMP, Gangs in OpenACC.
     *
     * @param The new amount of work groups.
     */
		void setThreadBlocks(int blocks ) { threadBlocks = blocks; }
    /**
     * @brief Sets the number of workers per work group.
     *
     * Warps in CUDA, Workers in OpenMP and OpenACC.
     *
     * @param The new amount of workers per work group.
     */
		void setThreadsPerBlocks(int threads) { threadsPerBlock = threads; }
	#endif

/****************************************************
 * 						Arrays used in kernels
****************************************************/
  /**
   * @brief Maps the RS_KERNEL_TYPE enum to the amount of bytes being moved.
   *
   * Initialized only after RSOpts::parseOpts.
   */
  double BYTES[NUM_KERNELS] = {
		// Original Kernels
		static_cast<double>(2 * sizeof(double)), // Copy
		static_cast<double>(2 * sizeof(double)), // Scale
		static_cast<double>(3 * sizeof(double)), // Add
		static_cast<double>(3 * sizeof(double)), // Triad
		// Gather Kernels
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t)))), // GATHER Copy
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t)))), // GATHER Scale
		static_cast<double>(((3 * sizeof(double)) + (2 * sizeof(ssize_t)))), // GATHER Add
		static_cast<double>(((3 * sizeof(double)) + (2 * sizeof(ssize_t)))), // GATHER Triad
		// Scatter Kernels
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t)))), // SCATTER Copy
		static_cast<double>(((2 * sizeof(double)) + (1 * sizeof(ssize_t)))), // SCATTER Scale
		static_cast<double>(((3 * sizeof(double)) + (1 * sizeof(ssize_t)))), // SCATTER Add
		static_cast<double>(((3 * sizeof(double)) + (1 * sizeof(ssize_t)))), // SCATTER Triad
		// Scatter-Gather Kernels
		static_cast<double>(((2 * sizeof(double)) + (2 * sizeof(ssize_t)))), // SG Copy
		static_cast<double>(((2 * sizeof(double)) + (2 * sizeof(ssize_t)))), // SG Scale
		static_cast<double>(((3 * sizeof(double)) + (3 * sizeof(ssize_t)))), // SG Add
		static_cast<double>(((3 * sizeof(double)) + (3 * sizeof(ssize_t)))), // SG Triad
		// Central Kernels
		static_cast<double>(2 * sizeof(double)), // Central Copy
		static_cast<double>(2 * sizeof(double)), // Central Scale
		static_cast<double>(3 * sizeof(double)), // Central Add
		static_cast<double>(3 * sizeof(double)), // Central Triad
  };

  /**
   * @brief Maps the RS_KERNEL_TYPE enum to the float ops incurred.
   *
   * Initialized only after RSOpts::parseOpts.
   */
  double FLOATOPS[NUM_KERNELS] = {
		// Original Kernels
		(double)0.0,                       // Copy
		1.0,             // Scale
		1.0,             // Add
		2.0,             // Triad
		// Gather Kernels
		(double)0.0,                       // GATHER Copy
		1.0,             // GATHER Scale
		1.0,             // GATHER Add
		2.0,             // GATHER Triad
		// Scatter Kernels
		(double)0.0,                       // SCATTER Copy
		1.0,             // SCATTER Scale
		1.0,             // SCATTER Add
		2.0,             // SCATTER Triad
		// Scatter-Gather Kernels
		(double)0.0,                       // SG Copy
		1.0,             // SG Scale
		1.0,             // SG Add
		2.0,             // SG Triad
		// Central Kernels
		(double)0.0,                       // CENTRAL Copy
		1.0,             // CENTRAL Scale
		1.0,             // CENTRAL Add
		2.0,             // CENTRAL Triad
  };

  /**
   * @brief Storage for the MBPS of benchmarks this invocation.
   */
	double MBPS[NUM_KERNELS]  = {0};
  /**
   * @brief Storage for the FLOPS of benchmarks this invocation.
   */
	double FLOPS[NUM_KERNELS] = {0};
  /**
   * @brief Storage for the runtimes of benchmarks this invocation.
   */
	double TIMES[NUM_KERNELS] = {0};
};

#endif /* _RSOPTS_H_ */
