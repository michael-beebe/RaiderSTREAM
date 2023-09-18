#include "RaiderSTREAM/RS_Opts.h"

Benchtype BenchTypeTable[] = {
// {  Name, Arg, Notes, KType, Enabled, ReqArq }
  { "seq", "", "Sequential",                RSBaseImpl::RS_NB, false, false },
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
  { "", "", "",                             RSBaseImpl::RS_NB, false, false }
};

// RSOpts Constructor
RSOpts::RSOpts()
  : STREAM_ARRAY_SIZE(1000000), NTIMES(10), STREAM_TYPE("double"), NTHREADS(1), NPROCS(1),
    LARGC(0), LARGV(NULL), is_help(false), is_list(false), is_verbose(false) {}
  
  
// RSOpts Destructor
RSOpts::~RSOpts() {}

RSBaseImpl::RSKernelType RSOpts::get_kernel_type() {
  unsigned Idx = 0;
  while (BenchTypeTable[Idx].KType != RSBaseImpl::RS_NB) {
    if (BenchTypeTable[Idx].Name == BenchName) {
      return BenchTypeTable[Idx].KType;
    }
    Idx++;
  }
  return RSBaseImpl::RS_NB;
}

vool RSOpts::enable_bench( std::string BenchName ) {
  unsigned Idx = 0;
  while (BenchTypeTable[Idx].KType != RSBaseImpl::RS_NB) {
    if (BenchTypeTable[Idx].Name == BenchName) {
      BenchTypeTable[Idx].Enabled = true;
      return true;
    }
    Idx++;
  }
  std::cout << "Invalid benchmark name: " << BenchName << std::endl;
  return false;
}

bool RSOpts::parse_opts( int argc, char **argv ) {
  LARGC = argc;
  LARGV = argv;
  for ( int i = 1; i < argc; i++ ) {
    std::string s(argv[i]);
    
    // get whether the user wants to prints the help message
    if ( (s=="-h") || (s=="-help") || (s=="--help") ) {
      is_help = true;
      print_help();
      return true;
    }
    // get the kernels the user wants to run
    // TODO: add ability to specify multiple but not all kernels
    else if( (s=="-k") || (s=="-kernel") || (s=="--kernel") ) {
      if (i+1 > (argc - 1)) {
        std::cout << "Error : --kernel requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i+1]);
      if( !EnableBench( P ) ) {
        std::cout << "Error : invalid argument for --kernel" << std::endl;
        return false;
      }
      i++;
    }
    // get the stream array size
    else if( (s=="-s" || (s=="-size") || (s=="--size")) ) {
      STREAM_ARRAY_SIZE = atoi(argv[i+1]);
      i++;
    }
    // get the number of times to run the benchmark
    else if( (s=="-n" || (s=="-ntimes") || (s=="--ntimes")) ) {
      NTIMES = atoi(argv[i+1]);
      i++;
    }
    // get the stream datatype
    else if( (s=="-d" || (s=="-dtype") || (s=="--dtype")) ) {
      STREAM_TYPE = argv[i+1];
      i++;
    }
    // // get the number of OpenMP threads
    // else if( (s=="-t" || (s=="-threads") || (s=="--threads")) ) {
    //   NTHREADS = atoi(argv[i+1]);
    //   i++;
    // }
    // get the number of PEs
    else if( (s=="-p" || (s=="-pes") || (s=="--pes")) ) {
      if (i+1 > (argc - 1)) {
        std::cout << "Error : --pes requires an argument" << std::endl;
        return false;
      }
      NPROCS = atoi(argv[i+1]);
      i++;
    }
    else if ( (s=="-l") || (s=="--list") || (s=="--list") ) {
      is_list = true;
      print_bench();
      return true;
    }
    else {
      std::cout << "Error : invalid argument: " << s << std::endl;
      return false;
    }
  }
}

// sanity check the options
if (STREAM_ARRAY_SIZE == 0) {
  std::cout << "Error : STREAM Array Size cannot be 0 " << std::endl;
  return false;
}
if (NTIMES == 0) {
  std::cout << "Error : NTIMES cannot be 0 " << std::endl;
  return false;
}
if (STREAM_TYPE != "double" && STREAM_TYPE != "float" && STREAM_TYPE != "int" && STREAM_TYPE != "long") {
  std::cout << "Error : STREAM_TYPE must be double, float, int, or long " << std::endl;
  return false;
}
if (NPROCS < 1) {
  std::cout << "Error : NPROCS must be greater than 0 " << std::endl;
  return false;
}


void RSOpts::print_bench() { // TODO: add in required options in the future
  std::cout << "===================================================================================" << std::endl;
  std::cout << "BENCHMARK | DESCRIPTION" << std::endl;
  std::cout << "===================================================================================" << std::endl;
  unsigned Idx = 0;
  while( BenchTypeTable[Idx].Name != "." ){
    std::cout << " - " << BenchTypeTable[Idx].Name;
    // if( BenchTypeTable[Idx].ReqArg ){
    //   std::cout << " | " << BenchTypeTable[Idx].Arg << " | ";
    // }else{
    //   std::cout << " | No Arg Required | ";
    // }
    std::cout << BenchTypeTable[Idx].Notes << std::endl;
    Idx++;
  }
  std::cout << "===================================================================================" << std::endl;
}

void RSOpts::print_help() {
  unsigned major = RS_VERSION_MAJOR;
  unsigned minor = RS_VERSION_MINOR;
  std::cout << "===================================================================================" << std::endl;
  std::cout << "RaiderSTREAM v" << major << "." << minor << std::endl;
  std::cout << "===================================================================================" << std::endl;
  std::cout << "Usage: ./RaiderSTREAM [OPTIONS]" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------" << std::endl;
  std::cout << "Options:" << std::endl;
  // std::cout << "-----------------------------------------------------------------------------------" << std::endl;
  std::cout << "  -h, --help, --help           Print this help message" << std::endl;
  std::cout << "  -l, --list, --list           List the benchmarks" << std::endl;
  std::cout << std::endl;
  std::cout << "  -k, --kernel, --kernel       Specify the kernel to run" << std::endl;
  std::cout << "  -s, --size, --size           Specify the size of the STREAM array" << std::endl;
  std::cout << "  -n, --ntimes, --ntimes       Specify the number of times to run the benchmark" << std::endl;
  std::cout << "  -d, --dtype, --dtype         Specify the datatype of the STREAM array" << std::endl;
  // std::cout << "  -t, --threads, --threads     Specify the number of OpenMP threads" << std::endl;
  std::cout << "  -p, --pes, --pes             Specify the number of PEs" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------" << std::endl;
}