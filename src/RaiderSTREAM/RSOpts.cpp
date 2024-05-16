//
// _RSOPTS_CPP_
//
// Copyright (C) 2022-2024 Texas Tech University
// All Rights Reserved
// michael.beebe@ttu.edu
//
// See LICENSE in the top level directory for licensing details
//

#include "RaiderSTREAM/RSOpts.h"
#include <algorithm>

BenchType BenchTypeTable[] = {
  // {  Name, Arg, Notes, KType, Enabled, ReqArq }
  { "seq_copy", "", "Sequential copy",      RSBaseImpl::RS_SEQ_COPY, false, false },
  { "seq_scale", "", "Sequential scale",    RSBaseImpl::RS_SEQ_SCALE, false, false },
  { "seq_add", "", "Sequential add",        RSBaseImpl::RS_SEQ_ADD, false, false },
  { "seq_triad", "", "Sequential triad",    RSBaseImpl::RS_SEQ_TRIAD, false, false },
  { "gather_copy", "", "Gather copy",       RSBaseImpl::RS_GATHER_COPY, false, false },
  { "gather_scale", "", "Gather scale",     RSBaseImpl::RS_GATHER_SCALE, false, false },
  { "gather_add", "", "Gather add",         RSBaseImpl::RS_GATHER_ADD, false, false },
  { "gather_triad", "", "Gather triad",     RSBaseImpl::RS_GATHER_TRIAD, false, false },
  { "scatter_copy", "", "Scatter copy",     RSBaseImpl::RS_SCATTER_COPY, false, false },
  { "scatter_scale", "", "Scatter scale",   RSBaseImpl::RS_SCATTER_SCALE, false, false },
  { "scatter_add", "", "Scatter add",       RSBaseImpl::RS_SCATTER_ADD, false, false },
  { "scatter_triad", "", "Scatter triad",   RSBaseImpl::RS_SCATTER_TRIAD, false, false },
  { "sg_copy", "", "Scatter-Gather copy",   RSBaseImpl::RS_SG_COPY, false, false },
  { "sg_scale", "", "Scatter-Gather scale", RSBaseImpl::RS_SG_SCALE, false, false },
  { "sg_add", "", "Scatter-Gather add",     RSBaseImpl::RS_SG_ADD, false, false },
  { "sg_triad", "", "Scatter-Gather triad", RSBaseImpl::RS_SG_TRIAD, false, false },
  { "central_copy", "", "Central copy",     RSBaseImpl::RS_CENTRAL_COPY, false, false },
  { "central_scale", "", "Central scale",   RSBaseImpl::RS_CENTRAL_SCALE, false, false },
  { "central_add", "", "Central add",       RSBaseImpl::RS_CENTRAL_ADD, false, false },
  { "central_triad", "", "Central triad",   RSBaseImpl::RS_CENTRAL_TRIAD, false, false },
  { "all", "", "All",                       RSBaseImpl::RS_ALL, false, false },
  { ".", "", ".",                           RSBaseImpl::RS_NB, false, false }
};

// RSOpts Constructor
RSOpts::RSOpts()
  : isHelp(false), isList(false), streamArraySize(1000000),
    numPEs(1), lArgc(0), lArgv(nullptr) {}

// RSOpts Destructor
RSOpts::~RSOpts() {}



// bool RSOpts::enableBenchmark(std::string benchName) {
//   unsigned Idx = 0;
//   while (BenchTypeTable[Idx].KType != RSBaseImpl::RS_NB) {
//     if (BenchTypeTable[Idx].Name == benchName) {
//       BenchTypeTable[Idx].Enabled = true;
//       return true;
//     }
//     Idx++;
//   }
//   std::cout << "Invalid benchmark name: " << benchName << std::endl;
//   return false;
// }

bool RSOpts::enableBenchmark(std::string benchName) {
  unsigned Idx = 0;
  std::transform(benchName.begin(), benchName.end(), benchName.begin(), ::tolower); // Convert benchName to lowercase
  while (BenchTypeTable[Idx].KType != RSBaseImpl::RS_NB) {
    std::string lowercaseName = BenchTypeTable[Idx].Name;
    std::transform(lowercaseName.begin(), lowercaseName.end(), lowercaseName.begin(), ::tolower); // Convert BenchTypeTable[Idx].Name to lowercase
    if (lowercaseName == benchName) {
      BenchTypeTable[Idx].Enabled = true;
      return true;
    }
    Idx++;
  }
  std::cout << "Invalid benchmark name: " << benchName << std::endl;
  return false;
}

bool RSOpts::parseOpts(int argc, char **argv) {
  lArgc = argc;
  lArgv = argv;
  for (int i = 1; i < argc; i++) {
    std::string s(argv[i]);

    // Check if the user wants to print the help message
    if ((s == "-h") || (s == "--help")) {
      isHelp = true;
      printHelp();
      return true;
    }
    else if ((s == "-l") || (s == "--list")) {
      isList = true;
      printBench();
      return true;
    }
    // Check for the kernels the user wants to run
    else if ((s == "-k") || (s == "--kernel")) {
      if (i + 1 > (argc - 1)) {
        std::cout << "Error: --kernel requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i + 1]);
      if (!enableBenchmark(P)) {
        std::cout << "Error: invalid argument for --kernel" << std::endl;
        return false;
      }
      // Set the kernelName after successfully enabling the benchmark
      kernelName = P;
      i++;
    }
    // Get the stream array size
    else if ((s == "-s") || (s == "--size")) {
      setStreamArraySize(atoi(argv[i + 1]));
      i++;
    }
    // Get the number of PEs
    else if ((s == "-np") || (s == "--pes")) {
      if (i + 1 > (argc - 1)) {
        std::cout << "Error: --pes requires an argument" << std::endl;
        return false;
      }
      // numPEs = atoi(argv[i + 1]);
      setNumPEs(atoi(argv[i + 1]));
      i++;
    }
    else {
      std::cout << "Error: invalid argument: " << s << std::endl;
      return false;
    }
  }

  // Sanity checks for the options
  if (streamArraySize == 0) {
    std::cout << "Error: STREAM Array Size cannot be 0" << std::endl;
    return false;
  }
  if (numPEs < 1) {
    std::cout << "Error: numPEs must be greater than 0" << std::endl;
    return false;
  }

  return true; // Options are valid
}

void RSOpts::printBench() {
  std::cout << "===================================================================================" << std::endl;
  std::cout << "BENCHMARK KERNEL | DESCRIPTION" << std::endl;
  std::cout << "===================================================================================" << std::endl;
  unsigned Idx = 0;
  while (BenchTypeTable[Idx].Name != "") {
    std::cout << "  " << BenchTypeTable[Idx].Name;
    if (BenchTypeTable[Idx].Name == "all") {
      std::cout << "\t\t| " << BenchTypeTable[Idx].Notes << std::endl; 
    }
    else {
      std::cout << "\t| " << BenchTypeTable[Idx].Notes << std::endl;
    }
    Idx++;
  }
  std::cout << "===================================================================================" << std::endl;
}

void RSOpts::printHelp() {
  unsigned major = RS_VERSION_MAJOR;
  unsigned minor = RS_VERSION_MINOR;
  std::cout << "===================================================================================" << std::endl;
  std::cout << " RaiderSTREAM v" << major << "." << minor << std::endl;
  std::cout << "===================================================================================" << std::endl;
  std::cout << " Usage: ./raiderstream [OPTIONS]" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------" << std::endl;
  std::cout << " Options:" << std::endl;
  std::cout << "  -h, --help                Print this help message" << std::endl;
  std::cout << "  -l, --list                List the benchmarks" << std::endl;
  std::cout << "  -k, --kernel              Specify the kernel to run" << std::endl;
  std::cout << "  -s, --size                Specify the size of the STREAM array" << std::endl;
  std::cout << "  -np, --pes                Specify the number of PEs" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------" << std::endl;
}
