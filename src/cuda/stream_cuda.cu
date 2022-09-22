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

// /*--------------------------------------------------------------------------------------
// - Initialize the STREAM arrays used in the kernels
// - Some compilers require an extra keyword to recognize the "restrict" qualifier.
// --------------------------------------------------------------------------------------*/
STREAM_TYPE __restrict__ *a;
STREAM_TYPE __restrict__ *b;
STREAM_TYPE __restrict__ *c;
STREAM_TYPE __restrict__ *d_a;
STREAM_TYPE __restrict__ *d_b;
STREAM_TYPE __restrict__ *d_c;

/*--------------------------------------------------------------------------------------
- Initialize IDX arrays (which will be used by gather/scatter kernels)
--------------------------------------------------------------------------------------*/
static ssize_t *IDX1;
static ssize_t *IDX2;
static ssize_t *IDX3;
static ssize_t *d_IDX1;
static ssize_t *d_IDX2;
static ssize_t *d_IDX3;

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

__global__ void stream_copy(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t stream_array_size) {
	ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[j];
}

__global__ void stream_scale(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[j] = scalar * d_c[j];
}

__global__ void stream_sum(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[j] + d_b[j];
}

__global__ void stream_triad(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[j] = d_b[j] + scalar * d_c[j];
}

void calculateTime(const cudaEvent_t& t0, const cudaEvent_t& t1, double& times[NUM_KERNELS][NTIMES], int round, Kernels kernel) {
	float ms = 0.0;
	cudaEventElapsedTime(&ms, t0, t1);
	times[kernel][round] = ms * 1E-3;
}

void executeSTREAM(STREAM_TYPE __restrict__   *a, STREAM_TYPE __restrict__   *b, STREAM_TYPE __restrict__   *c,
				   STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c,
				   ssize_t __restrict__  *d_IDX1, ssize_t __restrict__  *d_IDX2, ssize_t __restrict__  *d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size)
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		stream_copy<<<stream_array_size + 255/256, 256>>>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, COPY);

		cudaEventRecord(t0);
		stream_scale<<<stream_array_size + 255/256, 256>>>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCALE);

		cudaEventRecord(t0);
		stream_sum<<<stream_array_size + 255/256, 256>>>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SUM);

		cudaEventRecord(t0);
		stream_triad<<<stream_array_size + 255/256, 256>>>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, TRIAD);
	}
}

__global__ void gather_copy(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[d_IDX1[j]];
}

__global__ void gather_scale(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[j] = scalar * d_c[d_IDX2[j]];
}

__global__ void gather_sum(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[d_IDX1[j]] + d_b[d_IDX2[j]];
}

__global__ void gather_triad(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[j] = d_b[d_IDX1[j]] + scalar * d_c[d_IDX2[j]];
}

__global__ void scatter_copy(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[j];
}

__global__ void scatter_scale(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[d_IDX2[j]] = scalar * d_c[j];
}

__global__ void scatter_sum(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[j] + d_b[j];
}

__global__ void scatter_triad(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[d_IDX2[j]] = d_b[j] + scalar * d_c[j];
}

__global__ void sg_copy(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t __restrict__ *d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[d_IDX2[j]];
}

__global__ void sg_scale(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t __restrict__ *d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[d_IDX2[j]] = scalar * d_c[d_IDX1[j]];
}

__global__ void sg_sum(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t __restrict__ *d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[d_IDX2[j]] + d_b[d_IDX3[j]];
}

__global__ void sg_triad(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t __restrict__ *d_IDX1, ssize_t __restrict__ *d_IDX2, ssize_t __restrict__ *d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[d_IDX2[j]] = d_b[d_IDX3[j]] + scalar * d_c[d_IDX1[j]];
}

__global__ void central_copy(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t stream_array_size) {
	if(j < stream_array_size) d_c[0] = d_a[0];
}

__global__ void central_scale(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
	if(j < stream_array_size) d_b[0] = scalar * d_c[0];
}

__global__ void central_sum(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, ssize_t stream_array_size) {
	if(j < stream_array_size) d_c[0] = d_a[0] + d_b[0];
}

__global__ void central_triad(STREAM_TYPE __restrict__ *d_a, STREAM_TYPE __restrict__ *d_b, STREAM_TYPE __restrict__ *d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
	if(j < stream_array_size) d_a[0] = d_b[0] + scalar * d_c[0];
}

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

int main(int argc, char *argv[]) {
    ssize_t stream_array_size = 10000000; // Default stream_array_size is 10000000
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
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

	cudaMalloc(&d_a, sizeof(STREAM_TYPE) * stream_array_size);
	cudaMalloc(&d_b, sizeof(STREAM_TYPE) * stream_array_size);
	cudaMalloc(&d_c, sizeof(STREAM_TYPE) * stream_array_size);

	cudaMalloc(&d_IDX1, sizeof(ssize_t) * stream_array_size);
	cudaMalloc(&d_IDX2, sizeof(ssize_t) * stream_array_size);
	cudaMalloc(&d_IDX3, sizeof(ssize_t) * stream_array_size);

	double	bytes[NUM_KERNELS] = {
		// Original Kernels
		2 * sizeof(STREAM_TYPE) * stream_array_size, // Copy
		2 * sizeof(STREAM_TYPE) * stream_array_size, // Scale
		3 * sizeof(STREAM_TYPE) * stream_array_size, // Add
		3 * sizeof(STREAM_TYPE) * stream_array_size, // Triad
		// Gather Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // GATHER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // GATHER Scale
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // GATHER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // GATHER Triad
		// Scatter Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER Scale
		(((3 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER Add
		(((3 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * stream_array_size), // SCATTER Triad
		// Scatter-Gather Kernels
		(((2 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // SG copy
		(((2 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // SG Scale
		(((3 * sizeof(STREAM_TYPE)) + (3 * sizeof(ssize_t))) * stream_array_size), // SG Add
		(((3 * sizeof(STREAM_TYPE)) + (3 * sizeof(ssize_t))) * stream_array_size), // SG Triad
		// Central Kernels
		2 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Copy
		2 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Scale
		3 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Add
		3 * sizeof(STREAM_TYPE) * stream_array_size, // CENTRAL Triad
	};

	double   flops[NUM_KERNELS] = {
		// Original Kernels
		(int)0,                // Copy
		1 * stream_array_size, // Scale
		1 * stream_array_size, // Add
		2 * stream_array_size, // Triad
		// Gather Kernels
		(int)0,                // GATHER Copy
		1 * stream_array_size, // GATHER Scale
		1 * stream_array_size, // GATHER Add
		2 * stream_array_size, // GATHER Triad
		// Scatter Kernels
		(int)0,                // SCATTER Copy
		1 * stream_array_size, // SCATTER Scale
		1 * stream_array_size, // SCATTER Add
		2 * stream_array_size, // SCATTER Triad
        // Scatter-Gather Kernels
        (int)0,
		1 * stream_array_size, // SCATTER Scale
		1 * stream_array_size, // SCATTER Add
		2 * stream_array_size, // SCATTER Triad
		// Central Kernels
		(int)0,                // CENTRAL Copy
		1 * stream_array_size, // CENTRAL Scale
		1 * stream_array_size, // CENTRAL Add
		2 * stream_array_size, // CENTRAL Triad
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

	cudaMemcpy(d_IDX1, IDX1, sizeof(ssize_t) * stream_array_size);
	cudaMemcpy(d_IDX2, IDX2, sizeof(ssize_t) * stream_array_size);
	cudaMemcpy(d_IDX3, IDX3, sizeof(ssize_t) * stream_array_size);

/*--------------------------------------------------------------------------------------
    - Print initial info
--------------------------------------------------------------------------------------*/
    print_info1(BytesPerWord, stream_array_size);

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

    scalar = 3.0;

	executeSTREAM(a, b, c, d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, times, stream_array_size);
}