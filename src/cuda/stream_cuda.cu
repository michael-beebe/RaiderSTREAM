/*-----------------------------------------------------------------------*/
/* Program: RaiderSTREAM                                                 */
/* Original STREAM code developed by John D. McCalpin                    */
/* Programmers: Michael Beebe                                            */
/*              Brody Williams                                           */
/*              Pedro DaSilva                                            */
/*              Stephen Devaney                                          */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  4. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

#include "stream_cuda_output.cuh"
#include "stream_cuda_tuned.cuh"
#include "stream_cuda_validation.cuh"
// #include "stream_cuda_kernels.cuh"

using namespace std;

// /*--------------------------------------------------------------------------------------
// - Initialize the STREAM arrays used in the kernels
// - Some compilers require an extra keyword to recognize the "restrict" qualifier.
// --------------------------------------------------------------------------------------*/
STREAM_TYPE* __restrict__   a;
STREAM_TYPE* __restrict__   b;
STREAM_TYPE* __restrict__   c;
STREAM_TYPE* __restrict__ d_a;
STREAM_TYPE* __restrict__ d_b;
STREAM_TYPE* __restrict__ d_c;

/*--------------------------------------------------------------------------------------
- Initialize IDX arrays (which will be used by gather/scatter kernels)
--------------------------------------------------------------------------------------*/
static ssize_t*   IDX1;
static ssize_t*   IDX2;
static ssize_t*   IDX3;
static ssize_t* d_IDX1;
static ssize_t* d_IDX2;
static ssize_t* d_IDX3;

/*--------------------------------------------------------------------------------------
- Initialize arrays to store avgtime, maxime, and mintime metrics for each kernel.
- The default values are 0 for avgtime and maxtime.
- each mintime[] value needs to be set to FLT_MAX via a for loop inside main()
--------------------------------------------------------------------------------------*/
static double avgtime[NUM_KERNELS] = {0};
static double maxtime[NUM_KERNELS] = {0};
static double mintime[NUM_KERNELS];
static int is_validated[NUM_KERNELS] = {0};

/*--------------------------------------------------------------------------------------
- Function to populate the STREAM arrays
--------------------------------------------------------------------------------------*/
void init_arrays(ssize_t stream_array_size) {
	ssize_t j;
	
	#pragma omp parallel for private (j)
    for (j = 0; j < stream_array_size; j++) {
		a[j] = 2.0;
		b[j] = 2.0;
		c[j] = 0.0;
    }
}

__global__ void stream_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
	ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[j];
}

__global__ void stream_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[j] = scalar * d_c[j];
}

__global__ void stream_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[j] + d_b[j];
}

__global__ void stream_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[j] = d_b[j] + scalar * d_c[j];
}

void calculateTime(const cudaEvent_t& t0, const cudaEvent_t& t1, double times[NUM_KERNELS][NTIMES], int round, Kernels kernel) {
	float ms = 0.0;
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&ms, t0, t1);
	times[kernel][round] = ms * 1E-3;
}

void executeSTREAM(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		stream_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, COPY);

		cudaEventRecord(t0);
		stream_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCALE);

		cudaEventRecord(t0);
		stream_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SUM);

		cudaEventRecord(t0);
		stream_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	stream_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void gather_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[d_IDX1[j]];
}

__global__ void gather_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[j] = scalar * d_c[d_IDX2[j]];
}

__global__ void gather_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[d_IDX1[j]] + d_b[d_IDX2[j]];
}

__global__ void gather_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[j] = d_b[d_IDX1[j]] + scalar * d_c[d_IDX2[j]];
}

void executeGATHER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		gather_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_COPY);

		cudaEventRecord(t0);
		gather_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_SCALE);

		cudaEventRecord(t0);
		gather_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_SUM);

		cudaEventRecord(t0);
		gather_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	gather_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void scatter_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[j];
}

__global__ void scatter_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[d_IDX2[j]] = scalar * d_c[j];
}

__global__ void scatter_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[j] + d_b[j];
}

__global__ void scatter_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[d_IDX2[j]] = d_b[j] + scalar * d_c[j];
}

void executeSCATTER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		scatter_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_COPY);

		cudaEventRecord(t0);
		scatter_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_SCALE);

		cudaEventRecord(t0);
		scatter_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_SUM);

		cudaEventRecord(t0);
		scatter_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	scatter_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void sg_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[d_IDX2[j]];
}

__global__ void sg_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[d_IDX2[j]] = scalar * d_c[d_IDX1[j]];
}

__global__ void sg_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[d_IDX2[j]] + d_b[d_IDX3[j]];
}

__global__ void sg_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[d_IDX2[j]] = d_b[d_IDX3[j]] + scalar * d_c[d_IDX1[j]];
}

void executeSG(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		sg_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_COPY);

		cudaEventRecord(t0);
		sg_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_SCALE);

		cudaEventRecord(t0);
		sg_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_SUM);

		cudaEventRecord(t0);
		sg_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	sg_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void central_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
	d_c[0] = d_a[0];
}

__global__ void central_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
	d_b[0] = scalar * d_c[0];
}

__global__ void central_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
	d_c[0] = d_a[0] + d_b[0];
}

__global__ void central_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
	d_a[0] = d_b[0] + scalar * d_c[0];
}

void executeCENTRAL(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		central_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_COPY);

		cudaEventRecord(t0);
		central_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_SCALE);

		cudaEventRecord(t0);
		central_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_SUM);

		cudaEventRecord(t0);
		central_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	central_validation(stream_array_size, scalar, is_validated, a, b, c);
}

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

int main(int argc, char *argv[]) {
    ssize_t stream_array_size = 10000000; // Default stream_array_size is 10000000
    int			quantum, checktick();
    ssize_t		j;
    STREAM_TYPE		scalar = 3.0;
    double		t, times[NUM_KERNELS][NTIMES];
	double		t0,t1,tmin;

/*
    get stream_array_size at runtime
*/
    parse_opts(argc, argv, &stream_array_size);

/*
    Allocate the arrays on the host
*/
    a = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * stream_array_size+OFFSET);
    b = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * stream_array_size+OFFSET);
    c = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * stream_array_size+OFFSET);

	IDX1 = (ssize_t *) malloc(sizeof(ssize_t) * stream_array_size+OFFSET);
	IDX2 = (ssize_t *) malloc(sizeof(ssize_t) * stream_array_size+OFFSET);
    IDX3 = (ssize_t *) malloc(sizeof(ssize_t) * stream_array_size+OFFSET);

	cudaMalloc((void **) &d_a, sizeof(STREAM_TYPE) * stream_array_size);
	cudaMalloc((void **) &d_b, sizeof(STREAM_TYPE) * stream_array_size);
	cudaMalloc((void **) &d_c, sizeof(STREAM_TYPE) * stream_array_size);

	cudaMalloc((void **) &d_IDX1, sizeof(ssize_t) * stream_array_size);
	cudaMalloc((void **) &d_IDX2, sizeof(ssize_t) * stream_array_size);
	cudaMalloc((void **) &d_IDX3, sizeof(ssize_t) * stream_array_size);

	double	bytes[NUM_KERNELS] = {
		// Original Kernels
		(double) 2 * sizeof(STREAM_TYPE) * stream_array_size, // Copy
		(double) 2 * sizeof(STREAM_TYPE) * stream_array_size, // Scale
		(double) 3 * sizeof(STREAM_TYPE) * stream_array_size, // Add
		(double) 3 * sizeof(STREAM_TYPE) * stream_array_size, // Triad
		// Gather Kernels
		(double) (((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // GATHER copy
		(double) (((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // GATHER Scale
		(double) (((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // GATHER Add
		(double) (((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // GATHER Triad
		// Scatter Kernels
		(double) (((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER copy
		(double) (((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER Scale
		(double) (((3 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER Add
		(double) (((3 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER Triad
		// Scatter-Gather Kernels
		(double) (((2 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // SG copy
		(double) (((2 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // SG Scale
		(double) (((3 * sizeof(STREAM_TYPE)) + (3 * sizeof(ssize_t))) * stream_array_size), // SG Add
		(double) (((3 * sizeof(STREAM_TYPE)) + (3 * sizeof(ssize_t))) * stream_array_size), // SG Triad
		// Central Kernels
		(double) 2 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Copy
		(double) 2 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Scale
		(double) 3 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Add
		(double) 3 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Triad
	};

	double   flops[NUM_KERNELS] = {
		// Original Kernels
		(double) 0,                // Copy
		(double) 1 * stream_array_size, // Scale
		(double) 1 * stream_array_size, // Add
		(double) 2 * stream_array_size, // Triad
		// Gather Kernels
		(double) 0,                // GATHER Copy
		(double) 1 * stream_array_size, // GATHER Scale
		(double) 1 * stream_array_size, // GATHER Add
		(double) 2 * stream_array_size, // GATHER Triad
		// Scatter Kernels
		(double) 0,                // SCATTER Copy
		(double) 1 * stream_array_size, // SCATTER Scale
		(double) 1 * stream_array_size, // SCATTER Add
		(double) 2 * stream_array_size, // SCATTER Triad
        // Scatter-Gather Kernels
        (double) 0,
		(double) 1 * stream_array_size, // SCATTER Scale
		(double) 1 * stream_array_size, // SCATTER Add
		(double) 2 * stream_array_size, // SCATTER Triad
		// Central Kernels
		(double) 0,                // CENTRAL Copy
		(double) 1 * stream_array_size, // CENTRAL Scale
		(double) 1 * stream_array_size, // CENTRAL Add
		(double) 2 * stream_array_size, // CENTRAL Triad
	};

/*--------------------------------------------------------------------------------------
    - Set the mintime to default value (FLT_MAX) for each kernel, since we haven't executed
        any of the kernels or done any timing yet
--------------------------------------------------------------------------------------*/
    for (int i=0;i<NUM_KERNELS;i++) {
        mintime[i] = FLT_MAX;
    }

/*--------------------------------------------------------------------------------------
    - Initialize the idx arrays
	- Use the input .txt files to populate each array if the -DCUSTOM flag is enabled
	- If -DCUSTOM is not enabled, populate the IDX arrays with random values
--------------------------------------------------------------------------------------*/
#ifdef CUSTOM
	init_read_idx_array(IDX1, stream_array_size, "IDX1.txt");
	init_read_idx_array(IDX2, stream_array_size, "IDX2.txt");
	init_read_idx_array(IDX3, stream_array_size, "IDX2.txt");
#else
    srand(time(0));
    init_random_idx_array(IDX1, stream_array_size);
    init_random_idx_array(IDX2, stream_array_size);
    init_random_idx_array(IDX3, stream_array_size);
#endif

	cudaMemcpy(d_IDX1, IDX1, sizeof(ssize_t) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_IDX2, IDX2, sizeof(ssize_t) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_IDX3, IDX3, sizeof(ssize_t) * stream_array_size, cudaMemcpyHostToDevice);

/*--------------------------------------------------------------------------------------
    - Print initial info
--------------------------------------------------------------------------------------*/
    print_info1(stream_array_size);

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
#pragma omp master
	   {
    	    k = omp_get_num_threads();
    	    printf ("Number of Threads requested = %i\n",k);
        }
    }
#endif
#ifdef _OPENMP
	k = 0;
#pragma omp parallel
#pragma omp atomic
		k++;
    printf ("Number of Threads counted = %i\n",k);
#endif

/*--------------------------------------------------------------------------------------
    // Populate STREAM arrays
--------------------------------------------------------------------------------------*/
#pragma omp parallel for private (j)
    for (j=0; j<stream_array_size; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

// TODO: copy necessary data to device


/*--------------------------------------------------------------------------------------
    // Estimate precision and granularity of timer
--------------------------------------------------------------------------------------*/
	print_timer_granularity(quantum);

    t = mysecond();
#pragma omp parallel for private (j)
    for (j = 0; j < stream_array_size; j++) {
  		a[j] = 2.0E0 * a[j];
	}

    t = 1.0E6 * (mysecond() - t);

	print_info2(t, quantum);
	print_memory_usage(stream_array_size);

	executeSTREAM(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, stream_array_size, scalar, is_validated);
	executeGATHER(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, stream_array_size, scalar, is_validated);
	executeSCATTER(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, stream_array_size, scalar, is_validated);
	executeSG(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, stream_array_size, scalar, is_validated);
	executeCENTRAL(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, stream_array_size, scalar, is_validated);

/*--------------------------------------------------------------------------------------
	// Calculate results
--------------------------------------------------------------------------------------*/
    for (int k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<NUM_KERNELS; j++)
	    {
			avgtime[j] = avgtime[j] + times[j][k];
			mintime[j] = MIN(mintime[j], times[j][k]);
			maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

/*--------------------------------------------------------------------------------------
	// Print results table
--------------------------------------------------------------------------------------*/
    printf("Function\tBest Rate MB/s      Best FLOP/s\t   Avg time\t   Min time\t   Max time\n");
    for (j=0; j<NUM_KERNELS; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		if (j % 4 == 0) {
			printf(HLINE);
		}

        if (flops[j] == 0) {
            printf("%s%12.1f\t\t%s\t%11.6f\t%11.6f\t%11.6f\n",
                label[j].c_str(),                           // Kernel
                1.0E-06 * bytes[j]/mintime[j],      // MB/s
                "-",      // FLOP/s
                avgtime[j],                         // Avg Time
                mintime[j],                         // Min Time
                maxtime[j]);                        // Max time
        }
        else {
            printf("%s%12.1f\t%12.1f\t%11.6f\t%11.6f\t%11.6f\n",
                label[j].c_str(),                           // Kernel
                1.0E-06 * bytes[j]/mintime[j],      // MB/s
                1.0E-06 * flops[j]/mintime[j],      // FLOP/s
                avgtime[j],                         // Avg Time
                mintime[j],                         // Min Time
                maxtime[j]);                        // Max time
        }
    }
    printf(HLINE);

/*--------------------------------------------------------------------------------------
	// Validate results
--------------------------------------------------------------------------------------*/
	checkSTREAMresults(is_validated);
    printf(HLINE);

	free(a);
	free(b);
	free(c);

	free(IDX1);
	free(IDX2);
	free(IDX3);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	cudaFree(d_IDX1);
	cudaFree(d_IDX2);
	cudaFree(d_IDX3);

    return 0;
}