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

/** BenchType struct: defines an individual benchmark table entry */
// TODO: work in memory size, we will have user input that instead of streeam_array_size
typedef struct {
  const std::string Name;         // benchmark name
  std::string Arg;                // benchmark arguments
  std::string Notes;              // benchmark notes
  RSBaseImpl::RSKernelType KType; // benchmark kernel type
  bool Enabled;									  // is benchmark enabled
  bool ReqArq;										// does benchmark have required arguments
} BenchType;

class RSOpts {
private:
  bool is_help;                    // determines if the help option has been selected
  bool is_list;                    // list the benchmarks

  ssize_t STREAM_ARRAY_SIZE;      // stream array size
  int NTIMES;                     // number of times to run benchmark
  std::string STREAM_TYPE;        // stream datatype TODO: figure out how to parse this
  // int NTHREADS;                   // number of OpenMP threads
  int NPROCS;                     // Numer of ranks, PEs, etc.
  int LARGC;                      // main argc
  char **LARGV;                   // main argv

  void print_help();
  void print_version();
  void print_bench();
  bool enable_bench( std::string BenchName );

public:
  // default constructor
  RSOpts();
  
  // default destructor 
  ~RSOpts();
  
  
  // TODO: determine if this is the correct place for these two arrays
  double bytes[NUM_KERNELS] = {
		// Original Kernels
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Copy
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Scale
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Add
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Triad
		// Gather Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER Scale
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER Triad
		// Scatter Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER Scale
		(((3 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER Add
		(((3 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER Triad
		// Scatter-Gather Kernels
		(((2 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SG copy
		(((2 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SG Scale
		(((3 * sizeof(STREAM_TYPE)) + (3 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SG Add
		(((3 * sizeof(STREAM_TYPE)) + (3 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SG Triad
		// Central Kernels
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Central Copy
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Central Scale
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Central Add
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Central Triad
	};

	double flops[NUM_KERNELS] = {
		// Original Kernels
		(int)0,								 // Copy
		1 * STREAM_ARRAY_SIZE, // Scale
		1 * STREAM_ARRAY_SIZE, // Add
		2 * STREAM_ARRAY_SIZE, // Triad
		// Gather Kernels
		(int)0,								 // GATHER Copy
		1 * STREAM_ARRAY_SIZE, // GATHER Scale
		1 * STREAM_ARRAY_SIZE, // GATHER Add
		2 * STREAM_ARRAY_SIZE, // GATHER Triad
		// Scatter Kernels
		(int)0,								 // SCATTER Copy
		1 * STREAM_ARRAY_SIZE, // SCATTER Scale
		1 * STREAM_ARRAY_SIZE, // SCATTER Add
		2 * STREAM_ARRAY_SIZE, // SCATTER Triad
		// Scatter-Gather Kernels
		(int)0,								 // SG Copy
		1 * STREAM_ARRAY_SIZE, // SG Scale
		1 * STREAM_ARRAY_SIZE, // SG Add
		2 * STREAM_ARRAY_SIZE, // SG Triad
		// Central Kernels
		(int)0,								 // CENTRAL Copy
		1 * STREAM_ARRAY_SIZE, // CENTRAL Scale
		1 * STREAM_ARRAY_SIZE, // CENTRAL Add
		2 * STREAM_ARRAY_SIZE, // CENTRAL Triad
	};
  
  
  /****************
      Getters
  *****************/
  // Returns whether the help option was selected
  bool is_help() { return is_help; }

  // Returns whether the list options is selected
  bool is_list() { return is_list; }

  // Parse the input args
  bool parse_opts(int argc, char **argv);
  
  // Get the STREAM array size
  ssize_t get_STREAM_ARRAY_SIZE() { return STREAM_ARRAY_SIZE; }
  
  // Get the number of times to run the benchmark
  int get_num_times() { return NTIMES; }
  
  // // Get the number of OpenMP threads
  // int get_num_threads() { return NTHREADS; }
  
  // Get the number of ranks, PEs, etc.
  int get_num_procs() { return NPROCS; }
  
  /// Retrieve the argc value
  int get_argc() { return LARGC; }

  /// Retrieve the argv value
  char **get_argv() { return LARGV; }
};

#endif // _RSOPTS_H_