/*-----------------------------------------------------------------------*/
/* Program: RaiderSTREAM                                                 */
/* Original STREAM code developed by John D. McCalpin                    */
/* Programmers: Michael Beebe                                            */
/*              Brody Williams                                           */
/*              Pedro Barros                                             */
/*																		 */
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

using namespace std;

// /*--------------------------------------------------------------------------------------
// - Initialize the STREAM arrays used in the kernels
// - Some compilers require an extra keyword to recognize the "restrict" qualifier.
// --------------------------------------------------------------------------------------*/
STREAM_TYPE* __restrict__   a;
STREAM_TYPE* __restrict__   b;
STREAM_TYPE* __restrict__   c;
STREAM_TYPE* __restrict__ d_a[NUM_GPUS];
STREAM_TYPE* __restrict__ d_b[NUM_GPUS];
STREAM_TYPE* __restrict__ d_c[NUM_GPUS];

/*--------------------------------------------------------------------------------------
- Initialize IDX arrays (which will be used by gather/scatter kernels)
--------------------------------------------------------------------------------------*/
static ssize_t*   IDX1;
static ssize_t*   IDX2;
static ssize_t*   IDX3;
static ssize_t* d_IDX1[NUM_GPUS];
static ssize_t* d_IDX2[NUM_GPUS];
static ssize_t* d_IDX3[NUM_GPUS];

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
void init_arrays(ssize_t array_elements) {
	ssize_t j;
	
	#pragma omp parallel for private (j)
    for (j = 0; j < array_elements; j++) {
		a[j] = 2.0;
		b[j] = 2.0;
		c[j] = 0.0;
    }
}

__global__ void stream_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t array_elements) {
	ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[j] = d_a[j];
}

__global__ void stream_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_b[j] = scalar * d_c[j];
}

__global__ void stream_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[j] = d_a[j] + d_b[j];
}

__global__ void stream_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_a[j] = d_b[j] + scalar * d_c[j];
}

void calculateTime(double t0, double times[NUM_KERNELS][NTIMES], int round, Kernels kernel) {
	cudaDeviceSynchronize();
	MPI_Barrier(MPI_COMM_WORLD);
	times[kernel][round] = MPI_Wtime() - t0;
}

void executeSTREAM(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a[NUM_GPUS], STREAM_TYPE* __restrict__ d_b[NUM_GPUS], STREAM_TYPE* __restrict__ d_c[NUM_GPUS],
				   ssize_t* __restrict__  d_IDX1[NUM_GPUS], ssize_t* __restrict__  d_IDX2[NUM_GPUS], ssize_t* __restrict__  d_IDX3[NUM_GPUS],
				   double times[NUM_KERNELS][NTIMES], ssize_t elements_per_rank, ssize_t array_elements, STREAM_TYPE scalar, int is_validated[NUM_KERNELS], int myrank, int numranks)
{
	init_arrays(elements_per_rank);
	double t0;

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(d_a[deviceId], a, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_b[deviceId], b, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_c[deviceId], c, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
	}

	for(auto k = 0; k < NTIMES; k++) {
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			stream_copy<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], array_elements);
		}
		calculateTime(t0, times, k, COPY);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			stream_scale<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar, array_elements);
		}
		calculateTime(t0, times, k, SCALE);

		
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			stream_sum<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SUM);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			stream_triad<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar, array_elements);
		}
		calculateTime(t0, times, k, TRIAD);
	}

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(a + deviceId * array_elements, d_a[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(b + deviceId * array_elements, d_b[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(c + deviceId * array_elements, d_c[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
	}

	stream_validation(elements_per_rank, scalar, is_validated, a, b, c, myrank, numranks);
	MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void gather_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[j] = d_a[d_IDX1[j]];
}

__global__ void gather_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_b[j] = scalar * d_c[d_IDX2[j]];
}

__global__ void gather_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[j] = d_a[d_IDX1[j]] + d_b[d_IDX2[j]];
}

__global__ void gather_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_a[j] = d_b[d_IDX1[j]] + scalar * d_c[d_IDX2[j]];
}

void executeGATHER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a[NUM_GPUS], STREAM_TYPE* __restrict__ d_b[NUM_GPUS], STREAM_TYPE* __restrict__ d_c[NUM_GPUS],
				   ssize_t* __restrict__  d_IDX1[NUM_GPUS], ssize_t* __restrict__  d_IDX2[NUM_GPUS], ssize_t* __restrict__  d_IDX3[NUM_GPUS],
				   double times[NUM_KERNELS][NTIMES], ssize_t elements_per_rank, ssize_t array_elements, STREAM_TYPE scalar, int is_validated[NUM_KERNELS], int myrank, int numranks)
{
	init_arrays(elements_per_rank);
	double t0;

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(d_a[deviceId], a, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_b[deviceId], b, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_c[deviceId], c, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
	}

	for(auto k = 0; k < NTIMES; k++) {
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			gather_copy<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId],
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, GATHER_COPY);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			gather_scale<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar,
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, GATHER_SCALE);

		
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			gather_sum<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId],
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, GATHER_SUM);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			gather_triad<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar,
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, GATHER_TRIAD);
	}

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(a + deviceId * array_elements, d_a[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(b + deviceId * array_elements, d_b[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(c + deviceId * array_elements, d_c[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
	}

	gather_validation(elements_per_rank, scalar, is_validated, a, b, c, myrank, numranks);
	MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void scatter_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[d_IDX1[j]] = d_a[j];
}

__global__ void scatter_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_b[d_IDX2[j]] = scalar * d_c[j];
}

__global__ void scatter_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[d_IDX1[j]] = d_a[j] + d_b[j];
}

__global__ void scatter_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_a[d_IDX2[j]] = d_b[j] + scalar * d_c[j];
}

void executeSCATTER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				    STREAM_TYPE* __restrict__ d_a[NUM_GPUS], STREAM_TYPE* __restrict__ d_b[NUM_GPUS], STREAM_TYPE* __restrict__ d_c[NUM_GPUS],
				    ssize_t* __restrict__  d_IDX1[NUM_GPUS], ssize_t* __restrict__  d_IDX2[NUM_GPUS], ssize_t* __restrict__  d_IDX3[NUM_GPUS],
				    double times[NUM_KERNELS][NTIMES], ssize_t elements_per_rank, ssize_t array_elements, STREAM_TYPE scalar, int is_validated[NUM_KERNELS], int myrank, int numranks)
{
	init_arrays(elements_per_rank);
	double t0;

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(d_a[deviceId], a, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_b[deviceId], b, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_c[deviceId], c, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
	}

	for(auto k = 0; k < NTIMES; k++) {
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			scatter_copy<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId],
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SCATTER_COPY);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			scatter_scale<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar,
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SCATTER_SCALE);

		
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			scatter_sum<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId],
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SCATTER_SUM);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			scatter_triad<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar,
							d_IDX1[deviceId], d_IDX2[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SCATTER_TRIAD);
	}

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(a + deviceId * array_elements, d_a[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(b + deviceId * array_elements, d_b[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(c + deviceId * array_elements, d_c[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
	}

	scatter_validation(elements_per_rank, scalar, is_validated, a, b, c, myrank, numranks);
	MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void sg_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[d_IDX1[j]] = d_a[d_IDX2[j]];
}

__global__ void sg_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_b[d_IDX2[j]] = scalar * d_c[d_IDX1[j]];
}

__global__ void sg_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_c[d_IDX1[j]] = d_a[d_IDX2[j]] + d_b[d_IDX3[j]];
}

__global__ void sg_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t array_elements) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < array_elements) d_a[d_IDX2[j]] = d_b[d_IDX3[j]] + scalar * d_c[d_IDX1[j]];
}

void executeSG(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
			   STREAM_TYPE* __restrict__ d_a[NUM_GPUS], STREAM_TYPE* __restrict__ d_b[NUM_GPUS], STREAM_TYPE* __restrict__ d_c[NUM_GPUS],
			   ssize_t* __restrict__  d_IDX1[NUM_GPUS], ssize_t* __restrict__  d_IDX2[NUM_GPUS], ssize_t* __restrict__  d_IDX3[NUM_GPUS],
			   double times[NUM_KERNELS][NTIMES], ssize_t elements_per_rank, ssize_t array_elements, STREAM_TYPE scalar, int is_validated[NUM_KERNELS], int myrank, int numranks)
{
	init_arrays(elements_per_rank);
	double t0;

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(d_a[deviceId], a, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_b[deviceId], b, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_c[deviceId], c, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
	}

	for(auto k = 0; k < NTIMES; k++) {
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			sg_copy<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId],
							d_IDX1[deviceId], d_IDX2[deviceId], d_IDX3[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SG_COPY);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			sg_scale<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar,
							d_IDX1[deviceId], d_IDX2[deviceId], d_IDX3[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SG_SCALE);

		
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			sg_sum<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId],
							d_IDX1[deviceId], d_IDX2[deviceId], d_IDX3[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SG_SUM);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			sg_triad<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar,
							d_IDX1[deviceId], d_IDX2[deviceId], d_IDX3[deviceId], array_elements);
		}
		calculateTime(t0, times, k, SG_TRIAD);
	}

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(a + deviceId * array_elements, d_a[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(b + deviceId * array_elements, d_b[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(c + deviceId * array_elements, d_c[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
	}

	sg_validation(elements_per_rank, scalar, is_validated, a, b, c, myrank, numranks);
	MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void central_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t array_elements) {
	d_c[0] = d_a[0];
}

__global__ void central_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t array_elements) {
	d_b[0] = scalar * d_c[0];
}

__global__ void central_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t array_elements) {
	d_c[0] = d_a[0] + d_b[0];
}

__global__ void central_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t array_elements) {
	d_a[0] = d_b[0] + scalar * d_c[0];
}

void executeCENTRAL(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a[NUM_GPUS], STREAM_TYPE* __restrict__ d_b[NUM_GPUS], STREAM_TYPE* __restrict__ d_c[NUM_GPUS],
				   ssize_t* __restrict__  d_IDX1[NUM_GPUS], ssize_t* __restrict__  d_IDX2[NUM_GPUS], ssize_t* __restrict__  d_IDX3[NUM_GPUS],
				   double times[NUM_KERNELS][NTIMES], ssize_t elements_per_rank, ssize_t array_elements, STREAM_TYPE scalar, int is_validated[NUM_KERNELS], int myrank, int numranks)
{
	init_arrays(elements_per_rank);
	double t0;

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(d_a[deviceId], a, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_b[deviceId], b, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_c[deviceId], c, sizeof(STREAM_TYPE) * array_elements, cudaMemcpyHostToDevice) );
	}

	for(auto k = 0; k < NTIMES; k++) {
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			central_copy<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], array_elements);
		}
		calculateTime(t0, times, k, CENTRAL_COPY);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			central_scale<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar, array_elements);
		}
		calculateTime(t0, times, k, CENTRAL_SCALE);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			central_sum<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], array_elements);
		}
		calculateTime(t0, times, k, CENTRAL_SUM);

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
			gpuErrchk( cudaSetDevice(deviceId) );
			central_triad<<< (array_elements + 255)/256, 256 >>>(d_a[deviceId], d_b[deviceId], d_c[deviceId], scalar, array_elements);
		}
		calculateTime(t0, times, k, CENTRAL_TRIAD);
	}

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(a + deviceId * array_elements, d_a[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(b + deviceId * array_elements, d_b[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(c + deviceId * array_elements, d_c[deviceId], sizeof(STREAM_TYPE) * array_elements, cudaMemcpyDeviceToHost) );
	}

	central_validation(elements_per_rank, scalar, is_validated, a, b, c, myrank, numranks);
	MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

int main(int argc, char *argv[]) {
    ssize_t stream_array_size = 10000000; // Default stream_array_size is 10000000
    int			quantum, checktick();
    ssize_t		j;
	double		t, times[NUM_KERNELS][NTIMES];
    STREAM_TYPE		scalar = 0.42;
	double		*TimesByRank;
	STREAM_TYPE *AvgErrByRank;
	double		t0,t1,tmin;
	int         rc, numranks, myrank, k, i;

	rc = MPI_Init(NULL, NULL);
	t0 = MPI_Wtime();
    if (rc != MPI_SUCCESS) {
       printf("ERROR: MPI Initialization failed with return code %d\n",rc);
       exit(1);
    }
	// if either of these fail there is something really screwed up!
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if(!num_devices_check(myrank)) {
		printf(HLINE);
        printf("Rank %d\n", myrank);
        printf("NUM_GPUS should be set with -DNUM_GPUS=<number_of_gpus> when compiling. NUM_GPUS can not be greater than the number of available gpus\n");
        printf(HLINE);

        cudaError_t code = cudaErrorInvalidConfiguration;
        exit(code);
	}

/*
    get stream_array_size at runtime
*/
    parse_opts(argc, argv, &stream_array_size);

/*
    Allocate the arrays on the host
*/
	ssize_t elements_per_rank = stream_array_size / (ssize_t) numranks;
	ssize_t array_elements = elements_per_rank / NUM_GPUS;
	
	ssize_t array_size = elements_per_rank * sizeof(STREAM_TYPE);
	ssize_t array_alignment = 64;
	
	k = posix_memalign((void **)&a, array_alignment, array_size);
	if(k != 0) {
		printf("Rank %d: Allocation of array a failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
	}

	k = posix_memalign((void **)&b, array_alignment, array_size);
	if(k != 0) {
		printf("Rank %d: Allocation of array a failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
	}

	k = posix_memalign((void **)&c, array_alignment, array_size); 
	if(k != 0) {
		printf("Rank %d: Allocation of array a failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
	}

	IDX1 = (ssize_t *) malloc(sizeof(ssize_t) * array_elements);
	IDX2 = (ssize_t *) malloc(sizeof(ssize_t) * array_elements);
    IDX3 = (ssize_t *) malloc(sizeof(ssize_t) * array_elements);

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMalloc((void **) &d_a[deviceId],    sizeof(STREAM_TYPE) * elements_per_rank) );
		gpuErrchk( cudaMalloc((void **) &d_b[deviceId],    sizeof(STREAM_TYPE) * elements_per_rank) );
		gpuErrchk( cudaMalloc((void **) &d_c[deviceId],    sizeof(STREAM_TYPE) * elements_per_rank) );
		gpuErrchk( cudaMalloc((void **) &d_IDX1[deviceId], sizeof(ssize_t)     * array_elements) );
		gpuErrchk( cudaMalloc((void **) &d_IDX2[deviceId], sizeof(ssize_t)     * array_elements) );
		gpuErrchk( cudaMalloc((void **) &d_IDX3[deviceId], sizeof(ssize_t)     * array_elements) );
	}

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
	init_read_idx_array(IDX1, array_elements, "IDX1.txt");
	init_read_idx_array(IDX2, array_elements, "IDX2.txt");
	init_read_idx_array(IDX3, array_elements, "IDX2.txt");
#else
    srand(time(0));
    init_random_idx_array(IDX1, array_elements);
    init_random_idx_array(IDX2, array_elements);
    init_random_idx_array(IDX3, array_elements);
#endif

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaMemcpy(d_IDX1[deviceId], IDX1, sizeof(ssize_t) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_IDX2[deviceId], IDX2, sizeof(ssize_t) * array_elements, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_IDX3[deviceId], IDX3, sizeof(ssize_t) * array_elements, cudaMemcpyHostToDevice) );
	}

/*--------------------------------------------------------------------------------------
    - Print initial info
--------------------------------------------------------------------------------------*/
    if(myrank == 0)
		print_info1(numranks, elements_per_rank, array_elements);

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
    for (j=0; j<elements_per_rank; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

	if (myrank == 0) {
		// There are NUM_ARRAYS average error values for each rank (using STREAM_TYPE).
		AvgErrByRank = (double *) malloc(NUM_ARRAYS * sizeof(STREAM_TYPE) * numranks);
		if (AvgErrByRank == NULL) {
			printf("Ooops -- allocation of arrays to collect errors on MPI rank 0 failed\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		memset(AvgErrByRank, 0, NUM_ARRAYS * sizeof(STREAM_TYPE) * numranks);

		// There are NUM_KERNELS*NTIMES timing values for each rank (always doubles)
		TimesByRank = (double *) malloc(NUM_KERNELS * NTIMES * sizeof(double) * numranks);
		if (TimesByRank == NULL) {
			printf("Ooops -- allocation of arrays to collect timing data on MPI rank 0 failed\n");
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		memset(TimesByRank, 0, NUM_KERNELS * NTIMES * sizeof(double) * numranks);
	}

/*--------------------------------------------------------------------------------------
    // Estimate precision and granularity of timer
--------------------------------------------------------------------------------------*/
	if(myrank == 0)
		print_timer_granularity(quantum);

    t = MPI_Wtime();
#pragma omp parallel for private (j)
    for (j = 0; j < elements_per_rank; j++) {
  		a[j] = 2.0E0 * a[j];
	}

    t = 1.0E6 * (MPI_Wtime() - t);

	if(myrank == 0) {
		print_info2(t, quantum);
		print_memory_usage(stream_array_size);
	}

	executeSTREAM( a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, elements_per_rank, array_elements, scalar, is_validated, myrank, numranks);
	executeGATHER( a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, elements_per_rank, array_elements, scalar, is_validated, myrank, numranks);
	executeSCATTER(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, elements_per_rank, array_elements, scalar, is_validated, myrank, numranks);
	executeSG(     a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, elements_per_rank, array_elements, scalar, is_validated, myrank, numranks);
	executeCENTRAL(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, elements_per_rank, array_elements, scalar, is_validated, myrank, numranks);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(times, NUM_KERNELS * NTIMES, MPI_DOUBLE, TimesByRank, NUM_KERNELS * NTIMES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*--------------------------------------------------------------------------------------
	// Calculate results
--------------------------------------------------------------------------------------*/

	if (myrank == 0) {
		// for each iteration and each kernel, collect the minimum time across all MPI ranks
		// and overwrite the rank 0 "times" variable with the minimum so the original post-
		// processing code can still be used.
		for (k = 0; k < NTIMES; k++) {
			for (j = 0; j < NUM_KERNELS; j++) {
				tmin = 1.0e36;
				for (i = 0; i < numranks; i++) {
					// printf("DEBUG: Timing: iter %d, kernel %lu, rank %d, tmin %f, TbyRank %f\n",k,j,i,tmin,TimesByRank[NUM_KERNELS*NTIMES*i+j*NTIMES+k]);
					tmin = MIN(tmin, TimesByRank[NUM_KERNELS*NTIMES*i+j*NTIMES+k]);
				}
				// printf("DEBUG: Final Timing: iter %d, kernel %lu, final tmin %f\n",k,j,tmin);
				times[j][k] = tmin;
			}
		}

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
		printf("\n");
		printf(HLINE);
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
	}

/*--------------------------------------------------------------------------------------
	// Validate results
--------------------------------------------------------------------------------------*/
	if(myrank == 0) {
		checkSTREAMresults(is_validated);
		printf(HLINE);
	}

	free(a);
	free(b);
	free(c);

	free(IDX1);
	free(IDX2);
	free(IDX3);

	for(auto deviceId = 0; deviceId < NUM_GPUS; deviceId++) {
		gpuErrchk( cudaSetDevice(deviceId) );

		gpuErrchk( cudaFree(d_a[deviceId]) );
		gpuErrchk( cudaFree(d_b[deviceId]) );
		gpuErrchk( cudaFree(d_c[deviceId]) );
		gpuErrchk( cudaFree(d_IDX1[deviceId]) );
		gpuErrchk( cudaFree(d_IDX2[deviceId]) );
		gpuErrchk( cudaFree(d_IDX3[deviceId]) );
	}

    return 0;
}