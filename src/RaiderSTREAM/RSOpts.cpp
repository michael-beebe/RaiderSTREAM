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
  /* {  Name, Arg, Notes, KType, Enabled, ReqArq } */
  { "seq_copy", "", "Sequential Copy",      RSBaseImpl::RS_SEQ_COPY, false, false },
  { "seq_scale", "", "Sequential Scale",    RSBaseImpl::RS_SEQ_SCALE, false, false },
  { "seq_add", "", "Sequential Add",        RSBaseImpl::RS_SEQ_ADD, false, false },
  { "seq_triad", "", "Sequential Triad",    RSBaseImpl::RS_SEQ_TRIAD, false, false },
  { "gather_copy", "", "Gather Copy",       RSBaseImpl::RS_GATHER_COPY, false, false },
  { "gather_scale", "", "Gather Scale",     RSBaseImpl::RS_GATHER_SCALE, false, false },
  { "gather_add", "", "Gather Add",         RSBaseImpl::RS_GATHER_ADD, false, false },
  { "gather_triad", "", "Gather Triad",     RSBaseImpl::RS_GATHER_TRIAD, false, false },
  { "scatter_copy", "", "Scatter Copy",     RSBaseImpl::RS_SCATTER_COPY, false, false },
  { "scatter_scale", "", "Scatter Scale",   RSBaseImpl::RS_SCATTER_SCALE, false, false },
  { "scatter_add", "", "Scatter Add",       RSBaseImpl::RS_SCATTER_ADD, false, false },
  { "scatter_triad", "", "Scatter Triad",   RSBaseImpl::RS_SCATTER_TRIAD, false, false },
  { "sg_copy", "", "Scatter-Gather Copy",   RSBaseImpl::RS_SG_COPY, false, false },
  { "sg_scale", "", "Scatter-Gather Scale", RSBaseImpl::RS_SG_SCALE, false, false },
  { "sg_add", "", "Scatter-Gather Add",     RSBaseImpl::RS_SG_ADD, false, false },
  { "sg_triad", "", "Scatter-Gather Triad", RSBaseImpl::RS_SG_TRIAD, false, false },
  { "central_copy", "", "Central Copy",     RSBaseImpl::RS_CENTRAL_COPY, false, false },
  { "central_scale", "", "Central Scale",   RSBaseImpl::RS_CENTRAL_SCALE, false, false },
  { "central_add", "", "Central Add",       RSBaseImpl::RS_CENTRAL_ADD, false, false },
  { "central_triad", "", "Central Triad",   RSBaseImpl::RS_CENTRAL_TRIAD, false, false },
  { "all", "", "All",                       RSBaseImpl::RS_ALL, false, false },
  { ".", "", ".",                           RSBaseImpl::RS_NB, false, false }
};

/* RSOpts Constructor */
RSOpts::RSOpts()
  : isHelp(false), isList(false), streamArraySize(1000000),
    numPEs(1), lArgc(0), lArgv(nullptr) {}

/* RSOpts Destructor */ 
RSOpts::~RSOpts() {}


RSBaseImpl::RSKernelType RSOpts::getKernelTypeFromName(const std::string& kernelName) const {
  unsigned Idx = 0;
  while (BenchTypeTable[Idx].Name != ".") {
      if (BenchTypeTable[Idx].Name == kernelName) {
          return BenchTypeTable[Idx].KType;
      }
      Idx++;
  }
  return RSBaseImpl::RS_NB; // Return RS_NB if kernel name not found
}

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

    /* Check if the user wants to print the help message */
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
    /* Check for the kernels the user wants to run */
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
      setKernelName( P );
      i++;
    }
    /* Get the stream array size */
    else if ((s == "-s") || (s == "--size")) {
      setStreamArraySize(atoi(argv[i + 1]));
      i++;
    }
    /* Get the number of PEs */
    else if ((s == "-np") || (s == "--pes")) {
      if (i + 1 > (argc - 1)) {
        std::cout << "Error: --pes requires an argument" << std::endl;
        return false;
      }
      setNumPEs(atoi(argv[i + 1]));
      i++;
    }
#if _ENABLE_CUDA_ || _ENABLE_MPI_CUDA_ || _ENABLE_OMP_TARGET_
    else if ((s == "-b") || (s == "--blocks")) {
      if (i + 1 > (argc -1)) {
        std::cout << "Error: --blocks requires an argument" << std::endl;
        return false;
      }
      setThreadBlocks(atoi(argv[i + 1]));
      i++;
    }
    else if ((s == "-t") || (s == "--threads")) {
      if (i + 1 > (argc -1)) {
        std::cout << "Error: --threads requires an argument" << std::endl;
        return false;
      }
      setThreadsPerBlocks(atoi(argv[i + 1]));
      i++;
    }
#endif
    else {
      std::cout << "Error: invalid argument: " << s << std::endl;
      return false;
    }
  }

  /* Sanity checks for the options */
  if (streamArraySize == 0) {
    std::cout << "Error: STREAM Array Size cannot be 0" << std::endl;
    return false;
  }
  #ifdef _ENABLE_OMP_
  #else
    if (numPEs < 1) {
      std::cout << "Error: numPEs must be greater than or equal to 1" << std::endl;
      return false;
    }
  #endif

  return true; /* Options are valid */
}

void RSOpts::printOpts() {
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << "RaiderSTREAM Options:" << std::endl;
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << "Kernel Name: " << kernelName << std::endl;
  std::cout << "Kernel Type: " << static_cast<int>(getKernelType()) << std::endl;
  std::cout << "Stream Array Size: " << streamArraySize << std::endl;
  std::cout << "Number of PEs: " << numPEs << std::endl;
  char* ompNumThreads = getenv("OMP_NUM_THREADS");
  if (ompNumThreads != nullptr) { std::cout << "OMP_NUM_THREADS: " << ompNumThreads << std::endl; }
  else { std::cout << "OMP_NUM_THREADS: (not set)" << std::endl; }
}

void RSOpts::printBench() {
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << "BENCHMARK KERNEL | DESCRIPTION" << std::endl;
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
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
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
}

void RSOpts::printHelp() {
  unsigned major = RS_VERSION_MAJOR;
  unsigned minor = RS_VERSION_MINOR;
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << " Usage: ./raiderstream [OPTIONS]" << std::endl;
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
  std::cout << " Options:" << std::endl;
  std::cout << "  -h, --help                Print this help message" << std::endl;
  std::cout << "  -l, --list                List the benchmarks" << std::endl;
  std::cout << "  -k, --kernel              Specify the kernel to run" << std::endl;
  std::cout << "  -s, --size                Specify the size of the STREAM array" << std::endl;
  std::cout << "  -np, --pes                Specify the number of PEs" << std::endl;
#if _ENABLE_OMP_TARGET_ || _ENABLE_CUDA_ || _ENABLE_MPI_CUDA_
  std::cout << "  -b, --blocks              Specify the number of CUDA blocks or OMP teams" << std::endl;
  std::cout << "  -t, --threads             Specify the number of threads per block" << std::endl;
#endif
  std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
}

void RSOpts::printLogo() {
  std::cout << std::endl;
  std::cout << R"(
 _______           __       __                    ______  ________ _______  ________  ______  __       __ 
|       \         |  \     |  \                  /      \|        \       \|        \/      \|  \     /  \
| ▓▓▓▓▓▓▓\ ______  \▓▓ ____| ▓▓ ______   ______ |  ▓▓▓▓▓▓\\▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓\ ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓\ ▓▓\   /  ▓▓
| ▓▓__| ▓▓|      \|  \/      ▓▓/      \ /      \| ▓▓___\▓▓  | ▓▓  | ▓▓__| ▓▓ ▓▓__   | ▓▓__| ▓▓ ▓▓▓\ /  ▓▓▓
| ▓▓    ▓▓ \▓▓▓▓▓▓\ ▓▓  ▓▓▓▓▓▓▓  ▓▓▓▓▓▓\  ▓▓▓▓▓▓\\▓▓    \   | ▓▓  | ▓▓    ▓▓ ▓▓  \  | ▓▓    ▓▓ ▓▓▓▓\  ▓▓▓▓
| ▓▓▓▓▓▓▓\/      ▓▓ ▓▓ ▓▓  | ▓▓ ▓▓    ▓▓ ▓▓   \▓▓_\▓▓▓▓▓▓\  | ▓▓  | ▓▓▓▓▓▓▓\ ▓▓▓▓▓  | ▓▓▓▓▓▓▓▓ ▓▓\▓▓ ▓▓ ▓▓
| ▓▓  | ▓▓  ▓▓▓▓▓▓▓ ▓▓ ▓▓__| ▓▓ ▓▓▓▓▓▓▓▓ ▓▓     |  \__| ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓_____| ▓▓  | ▓▓ ▓▓ \▓▓▓| ▓▓
| ▓▓  | ▓▓\▓▓    ▓▓ ▓▓\▓▓    ▓▓\▓▓     \ ▓▓      \▓▓    ▓▓  | ▓▓  | ▓▓  | ▓▓ ▓▓     \ ▓▓  | ▓▓ ▓▓  \▓ | ▓▓
 \▓▓   \▓▓ \▓▓▓▓▓▓▓\▓▓ \▓▓▓▓▓▓▓ \▓▓▓▓▓▓▓\▓▓       \▓▓▓▓▓▓    \▓▓   \▓▓   \▓▓\▓▓▓▓▓▓▓▓\▓▓   \▓▓\▓▓      \▓▓
  )";
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
}

// void RSOpts::printLogo() {
//   std::cout << std::setfill('=') << std::setw(110) << "=" << std::endl;
//   std::cout << R"(
//          ___        _     _            ___  _____  ___  ___  ___  __  __ 
//         | _ \ __ _ (_) __| | ___  _ _ / __||_   _|| _ \| __|/   \|  \/  |
//         |   // _` || |/ _` |/ -_)| '_|\__ \  | |  |   /| _| | - || |\/| |
//         |_|_\\__/_||_|\__/_|\___||_|  |___/  |_|  |_|_\|___||_|_||_|  |_|
//   )";
//   std::cout << std::endl;
//   std::cout << std::endl;
//   std::cout << std::setfill('=') << std::setw(110) << "=" << std::endl;
// }

// void RSOpts::printLogo() {
//   std::cout << std::setfill('=') << std::setw(110) << "=" << std::endl;
//   std::cout << R"(
//               ╔═══╗         ╔╗       ╔═══╗╔════╗╔═══╗╔═══╗╔═══╗╔═╗╔═╗
//               ║╔═╗║         ║║       ║╔═╗║║╔╗╔╗║║╔═╗║║╔══╝║╔═╗║║║╚╝║║
//               ║╚═╝║╔══╗ ╔╗╔═╝║╔══╗╔═╗║╚══╗╚╝║║╚╝║╚═╝║║╚══╗║║ ║║║╔╗╔╗║
//               ║╔╗╔╝╚ ╗║ ╠╣║╔╗║║╔╗║║╔╝╚══╗║  ║║  ║╔╗╔╝║╔══╝║╚═╝║║║║║║║
//               ║║║╚╗║╚╝╚╗║║║╚╝║║║═╣║║ ║╚═╝║ ╔╝╚╗ ║║║╚╗║╚══╗║╔═╗║║║║║║║
//               ╚╝╚═╝╚═══╝╚╝╚══╝╚══╝╚╝ ╚═══╝ ╚══╝ ╚╝╚═╝╚═══╝╚╝ ╚╝╚╝╚╝╚╝
//   )";
//   std::cout << std::endl;
//   std::cout << std::endl;
//   std::cout << std::setfill('=') << std::setw(110) << "=" << std::endl;
// }
                                                                                                          
                                                                                                          
                                                                                                          
