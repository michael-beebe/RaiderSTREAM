#include <omp.h>
#include <stdint.h>

/**************************************************
 * @brief Copies data from one stream to another.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void seq_copy(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[j] = a[j];
  t1 = mysecond();
  times[COPY][k] = t1 - t0;
}

/**************************************************
 * @brief Scales data in a stream.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void seq_scale(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    b[j] = scalar * c[j];
  t1 = mysecond();
  times[SCALE][k] = t1 - t0;
}

/**************************************************
 * @brief Adds data from two streams.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void seq_add(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[j] = a[j] + b[j];
  t1 = mysecond();
  times[SUM][k] = t1 - t0;
}

/**************************************************
 * @brief Performs triad operation on stream data.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void seq_triad(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    a[j] = b[j] + scalar * c[j];
  t1 = mysecond();
  times[TRIAD][k] = t1 - t0;
}

/**************************************************
 * @brief Copies data using gather operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void gather_copy(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[j] = a[IDX1[j]];
  t1 = mysecond();
  times[GATHER_COPY][k] = t1 - t0;
}

/**************************************************
 * @brief Scales data using gather operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void gather_scale(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    b[j] = scalar * c[IDX1[j]];
  t1 = mysecond();
  times[GATHER_SCALE][k] = t1 - t0;  
}

/**************************************************
 * @brief Adds data using gather operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void gather_add(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t *IDX2,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[j] = a[IDX1[j]] + b[IDX2[j]];
  t1 = mysecond();
  times[GATHER_SUM][k] = t1 - t0;
}

/**************************************************
 * @brief Performs triad operation using gather.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void gather_triad(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t *IDX2,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    a[j] = b[IDX1[j]] + scalar * c[IDX2[j]];
  t1 = mysecond();
  times[GATHER_TRIAD][k] = t1 - t0;
}

/**************************************************
 * @brief Copies data using scatter operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void scatter_copy(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[IDX1[j]] = a[j];
  t1 = mysecond();
  times[SCATTER_COPY][k] = t1 - t0;
}

/**************************************************
 * @brief Scales data using scatter operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void scatter_scale(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    b[IDX1[j]] = scalar * c[j];
  t1 = mysecond();
  times[SCATTER_SCALE][k] = t1 - t0;  
}

/**************************************************
 * @brief Adds data using scatter operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void scatter_add(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[IDX1[j]] = a[j] + b[j];
  t1 = mysecond();
  times[SCATTER_SUM][k] = t1 - t0;
}

/**************************************************
 * @brief Performs triad operation using scatter.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void scatter_triad(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    a[IDX1[j]] = b[j] + scalar * c[j];
  t1 = mysecond();
  times[SCATTER_TRIAD][k] = t1 - t0;
}

/**************************************************
 * @brief Copies data using scatter-gather operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void sg_copy(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t *IDX2,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[IDX1[j]] = a[IDX2[j]];
  t1 = mysecond();
  times[SG_COPY][k] = t1 - t0;
}

/**************************************************
 * @brief Scales data using scatter-gather operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void sg_scale(
  double *a,
  double *b,
  double *c,
  ssize_t *IDX1,
  ssize_t *IDX2,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    b[IDX2[j]] = scalar * c[IDX1[j]];
  t1 = mysecond();
  times[SG_SCALE][k] = t1 - t0;  
}

/**************************************************
 * @brief Adds data using scatter-gather operation.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
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
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[IDX1[j]] = a[IDX2[j]] + b[IDX3[j]];
  t1 = mysecond();
  times[SG_SUM][k] = t1 - t0;
}

/**************************************************
 * @brief Performs triad operation using scatter-gather.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
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
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    a[IDX2[j]] = b[IDX3[j]] + scalar * c[IDX1[j]];
  t1 = mysecond();
  times[SG_TRIAD][k] = t1 - t0;
}

/**************************************************
 * @brief Copies data using a central location.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void central_copy(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[0] = a[0];
  t1 = mysecond();
  times[CENTRAL_COPY][k] = t1 - t0;
}

/**************************************************
 * @brief Scales data using a central location.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void central_scale(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    b[0] = scalar * c[0];
  t1 = mysecond();
  times[CENTRAL_SCALE][k] = t1 - t0;
}

/**************************************************
 * @brief Adds data using a central location.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void central_add(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    c[0] = a[0] + b[0];
  t1 = mysecond();
  times[CENTRAL_SUM][k] = t1 - t0;
}

/**************************************************
 * @brief Performs triad operation using a central location.
 * 
 * @param stream_array_size Size of the stream array.
 * @param times 2D array to store kernel execution times.
 * @param k Kernel index.
 * @param scalar Scalar value for operations.
 **************************************************/
void central_triad(
  double *a,
  double *b,
  double *c,
  ssize_t stream_array_size,
  double times[NUM_KERNELS][NTIMES],
  int k,
  double scalar)
{
  double t0, t1;
  ssize_t j;

  t0 = mysecond();
  #pragma omp parallel for
  for (j = 0; j < stream_array_size; j++)
    a[0] = b[0] + scalar * c[0];
  t1 = mysecond();
  times[CENTRAL_TRIAD][k] = t1 - t0;
}