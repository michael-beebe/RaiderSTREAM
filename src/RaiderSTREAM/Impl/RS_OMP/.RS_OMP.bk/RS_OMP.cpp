#include "RS_OMP.h"

#ifdef _RS_OMP_H_

RS_OMP::RS_OMP(
  RSBaseImpl::RSBenchType bench_type,
  RSBaseImpl::RSKernelType kernel_type
) :
  RSBaseImpl("OMP", bench_type, kernel_type),
  a(nullptr),
  b(nullptr),
  c(nullptr),
  IDX1(nullptr),
  IDX2(nullptr),
  IDX3(nullptr),
  avgtime(nullptr),
  maxtime(nullptr),
  mintime(nullptr),
  scalar(3.0),
  STREAM_ARRAY_SIZE(1000000),
  NTIMES(10),
  NumThreads(1),
  NumProcs(1)
  // TODO: find what else we need
{}
RS_OMP::~RS_OMP(){}

bool RS_OMP::alocate_data() {
  a = (double *)malloc(sizeof(double) * STREAM_ARRAY_SIZE);
  b = (double *)malloc(sizeof(double) * STREAM_ARRAY_SIZE);
  c = (double *)malloc(sizeof(double) * STREAM_ARRAY_SIZE);
  IDX1 = (ssize_t *)malloc(sizeof(ssize_t) * STREAM_ARRAY_SIZE);
  IDX2 = (ssize_t *)malloc(sizeof(ssize_t) * STREAM_ARRAY_SIZE);
  IDX3 = (ssize_t *)malloc(sizeof(ssize_t) * STREAM_ARRAY_SIZE);
  avgtime = (double *)malloc(sizeof(double) * NUM_KERNELS);
  maxtime = (double *)malloc(sizeof(double) * NUM_KERNELS);
  mintime = (double *)malloc(sizeof(double) * NUM_KERNELS);
  return true;
}

bool RS_OMP::free_data() {
  if ( a ) { free( a ); }
  if ( b ) { free( b ); }
  if ( c ) { free( c ); }
  if ( IDX1 ) { free( IDX1 ); }
  if ( IDX2 ) { free( IDX2 ); }
  if ( IDX3 ) { free( IDX3 ); }
  if ( avgtime ) { free( avgtime ); }
  if ( maxtime ) { free( maxtime ); }
  if ( mintime ) { free( mintime ); }
  return true;
}

bool RS_OMP::execute(/*TODO: fill in args*/){
  RSBaseImpl::RSKernelType kernel_type = get_kernel_type();
  double start_time = 0.;
  double end_time = 0.;

  // TODO: Run desired kernels
  switch ( kernel_type ) {
    case RSBaseImpl::RS_ALL:
      // TODO: run all kernels
      break;
    case RSBaseImpl::RS_SEQ_COPY:
      start_time = this->my_second();
      seq_copy(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SEQ_SCALE:
      start_time = this->my_second();
      seq_scale(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SEQ_ADD:
      start_time = this->my_second();
      seq_add(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SEQ_TRIAD:
      start_time = this->my_second();
      seq_triad(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_GATHER_COPY:
      start_time = this->my_second();
      gather_copy(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_GATHER_SCALE:
      start_time = this->my_second();
      gather_scale(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_GATHER_ADD:
      start_time = this->my_second();
      gather_add(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_GATHER_TRIAD:
      start_time = this->my_second();
      gather_triad(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SCATTER_COPY:
      start_time = this->my_second();
      scatter_copy(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SCATTER_SCALE:
      start_time = this->my_second();
      scatter_scale(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SCATTER_ADD:
      start_time = this->my_second();
      scatter_add(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SCATTER_TRIAD:
      start_time = this->my_second();
      scatter_triad(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SG_COPY:
      start_time = this->my_second();
      sg_copy(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SG_SCALE:
      start_time = this->my_second();
      sg_scale(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SG_ADD:
      start_time = this->my_second();
      sg_add(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_SG_TRIAD:
      start_time = this->my_second();
      sg_triad(a, b, c, IDX1, IDX2, IDX3, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_CENTRAL_COPY:
      start_time = this->my_second();
      central_copy(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_CENTRAL_SCALE:
      start_time = this->my_second();
      central_scale(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_CENTRAL_ADD:
      start_time = this->my_second();
      central_add(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    case RSBaseImpl::RS_CENTRAL_TRIAD:
      start_time = this->my_second();
      central_triad(a, b, c, STREAM_ARRAY_SIZE, times, k, scalar);
      end_time = this->my_second();
      break;
    default:
      this->print_help(); // FIXME: maybe do something else here
  
  // TODO: gather results
  
  // TODO: validate results
  
  
  return true;
}

#endif // _RS_OMP_H_