#ifdef _ENABLE_OMP_
#ifndef _RS_OMP_H_

#include <cstdlib>
#include <omp.h>
#include <ctime>

#include "RaiderSTREAM/RSBaseImpl.h"

extern "C" { // TODO: updates the arguments here
// sequential copy
  void seq_copy(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sequential scale
  void seq_scale(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sequential add
  void seq_add(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sequential triad
  void seq_triad(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // gather copy
  void gather_copy(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // gather scale
  void gather_scale(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // gather add
  void gather_add(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // gather triad
  void gather_triad(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // scatter copy
  void scatter_copy(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // scatter scale
  void scatter_scale(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // scatter add
  void scatter_add(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // scatter triad
  void scatter_triad(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sg copy
  void sg_copy(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sg scale
  void sg_scale(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sg add
  void sg_add(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t *IDX3,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // sg triad
  void sg_triad(
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t *IDX3,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES],
    int k,
    double scalar
  );
  
  // central copy
  void central_copy(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES]
    int k,
    double scalar
  );
  
  // central scale
  void central_scale(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES]
    int k,
    double scalar
  );
  
  // central add
  void central_add(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES]
    int k,
    double scalar
  );
  
  // central triad
  void central_triad(
    double *a,
    double *b,
    double *c,
    ssize_t stream_array_size,
    double times[NUM_KERNELS][NTIMES]
    int k,
    double scalar
  );
}

class RS_OMP : public RSBaseImpl {
private:
  double *a;                          // STREAM array
  double *b;                          // STREAM array
  double *c;                          // STREAM array
  double *IDX1;                       // random index array
  double *IDX2;                       // random index array
  double *IDX3;                       // random index array
  ssize_t STREAM_ARRAY_SIZE;          // size of the STREAM arrays
  double times[NUM_KERNELS][NTIMES];  // Array for storing the timings
  int k;                              // index of the kernel FIXME:

public:
  // RaiderSTREAM OpenMP constructor
  RS_OMP(
    RSBaseImpl::RSBenchType BenchType
  );
  // RaiderSTREAM OpenMP destructor
  ~RS_OMP();
  
  // RaiderSTREAM OpenMP data allocation function
  virtual bool allocate_data(/*TODO: fill these in*/) override;
  
  // RaiderSTREAM OpenMP execute function
  virtual bool execute(/*fill in these args*/) override;

  virtual bool free_data() override;
};

#endif // _RS_OMP_H_
#endif // _ENABLE_OMP_