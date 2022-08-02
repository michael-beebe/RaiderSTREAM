/*-----------------------------------------------------------------------*/
/* Program: RaiderSTREAM                                                 */
/* Original STREAM code developed by John D. McCalpin                    */
/* Programmers: Michael Beebe                                            */
/*              Brody Williams                                           */
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
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*           "tuned STREAM benchmark results"                            */
/*           "based on a variant of the STREAM benchmark code"           */
/*         Other comparable, clear, and reasonable labelling is          */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

# define _XOPEN_SOURCE 600

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <string.h>
# include <limits.h>
# include <sys/time.h>
# include <time.h>

# include "mpi.h"

/*-----------------------------------------------------------------------
 * INSTRUCTIONS:
 *
 *	1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 *
 *      Version 5.10 increases the default array size from 2 million
 *          elements to 10 million elements in response to the increasing
 *          size of L3 caches.  The new default size is large enough for caches
 *          up to 20 MB.
 *      Version 5.10 changes the loop index variables from "register int"
 *          to "ssize_t", which allows array indices >2^32 (4 billion)
 *          on properly configured 64-bit systems.  Additional compiler options
 *          (such as "-mcmodel=medium") may be required for large memory runs.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                gcc -O -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream.100M
 *          will override the default size of 10M with a new size of 100M elements
 *          per array.
 */

// ----------------------- !!! NOTE CHANGE IN DEFINITION !!! ------------------
// For the MPI version of STREAM, the three arrays with this many elements
// each will be *distributed* across the MPI ranks.
//
// Be careful when computing the array size needed for a particular target
// system to meet the minimum size requirement to ensure overflowing the caches.
//
// Example:
//    Assume 4 nodes with two Intel Xeon E5-2680 processors (20 MiB L3) each.
//    The *total* L3 cache size is 4*2*20 = 160 MiB, so each array must be
//    at least 640 MiB, or at least 80 million 8 Byte elements.
// Note that it does not matter whether you use one MPI rank per node or
//    16 MPI ranks per node -- only the total array size and the total
//    cache size matter.
//
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

/*  
 *   2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif


// Make the scalar coefficient modifiable at compile time.
// The old value of 3.0 cause floating-point overflows after a relatively small
// number of iterations.  The new default of 0.42 allows over 2000 iterations for
// 32-bit IEEE arithmetic and over 18000 iterations for 64-bit IEEE arithmetic.
// The growth in the solution can be eliminated (almost) completely by setting
// the scalar value to 0.41421445, but this also means that the error checking
// code no longer triggers an error if the code does not actually execute the
// correct number of iterations!
#ifndef SCALAR
#define SCALAR 0.42
#endif


// ----------------------- !!! NOTE CHANGE IN DEFINITION !!! ------------------
// The OFFSET preprocessor variable is not used in this version of the benchmark.
// The user must change the code at or after the "posix_memalign" array allocations
//    to change the relative alignment of the pointers.
// ----------------------- !!! NOTE CHANGE IN DEFINITION !!! ------------------
#ifndef OFFSET
#   define OFFSET	0
#endif


/*
 *	3) Compile the code with optimization.  Many compilers generate
 *       unreasonably bad code before the optimizer tightens things up.
 *     If the results are unreasonably good, on the other hand, the
 *       optimizer might be too smart for me!
 *
 *     For a simple single-core version, try compiling with:
 *            cc -O stream.c -o stream
 *     This is known to work on many, many systems....
 *
 *     To use multiple cores, you need to tell the compiler to obey the OpenMP
 *       directives in the code.  This varies by compiler, but a common example is
 *            gcc -O -fopenmp stream.c -o stream_omp
 *       The environment variable OMP_NUM_THREADS allows runtime control of the
 *         number of threads/cores used when the resulting "stream_omp" program
 *         is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1) above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes the
 *       code to call separate functions to execute each kernel.  Trivial versions
 *       of these functions are provided, but they are *not* tuned -- they just
 *       provide predefined interfaces to be replaced with tuned code.
 *
 *
 *	4) Optional: Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include info that will help me understand:
 *		a) the computer hardware configuration (e.g., processor model, memory type)
 *		b) the compiler name/version and compilation flags
 *      c) any run-time information (such as OMP_NUM_THREADS)
 *		d) all of the output from the test case.
 *
 * Thanks!
 *
 *-----------------------------------------------------------------------*/

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

/*--------------------------------------------------------------------------------------
- Specifies the total number of benchmark kernels
- This is important as it is used throughout the benchmark code
--------------------------------------------------------------------------------------*/
# ifndef NUM_KERNELS
# define NUM_KERNELS 12
# endif

/*--------------------------------------------------------------------------------------
- Specifies the total number of stream arrays used in the main loop
--------------------------------------------------------------------------------------*/
# ifndef NUM_ARRAYS
# define NUM_ARRAYS 3
# endif

/*--------------------------------------------------------------------------------------
- Initialize the STREAM arrays used in the kernels
- Some compilers require an extra keyword to recognize the "restrict" qualifier.
--------------------------------------------------------------------------------------*/
STREAM_TYPE * restrict a, * restrict b, * restrict c;

/*--------------------------------------------------------------------------------------
- Initialize idx arrays (which will be used by gather/scatter kernels)
--------------------------------------------------------------------------------------*/
static int a_idx[STREAM_ARRAY_SIZE];
static int b_idx[STREAM_ARRAY_SIZE];
static int c_idx[STREAM_ARRAY_SIZE];

/*--------------------------------------------------------------------------------------
-
--------------------------------------------------------------------------------------*/
size_t		array_elements, array_bytes, array_alignment;

/*--------------------------------------------------------------------------------------
- Initialize arrays to store avgtime, maxime, and mintime metrics for each kernel.
- The default values are 0 for avgtime and maxtime.
- each mintime[] value needs to be set to FLT_MAX via a for loop inside main()
--------------------------------------------------------------------------------------*/
static double avgtime[NUM_KERNELS] = {0};
static double maxtime[NUM_KERNELS] = {0};
static double mintime[NUM_KERNELS];

/*--------------------------------------------------------------------------------------
- Initialize array to store labels for the benchmark kernels.
--------------------------------------------------------------------------------------*/
static char	*label[NUM_KERNELS] = {
    "Copy:\t\t", "Scale:\t\t",
    "Add:\t\t", "Triad:\t\t",
	"GATHER Copy:\t", "GATHER Scale:\t",
	"GATHER Add:\t", "GATHER Triad:\t",
	"SCATTER Copy:\t", "SCATTER Scale:\t",
	"SCATTER Add:\t", "SCATTER Triad:\t"
};

/*--------------------------------------------------------------------------------------
- Initialize array for storing the number of bytes that needs to be counted for
  each benchmark kernel
--------------------------------------------------------------------------------------*/
static double bytes[NUM_KERNELS] = {
	// Original Kernels
	2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Copy
	2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Scale
	3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Add
	3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Triad
	// Gather Kernels
	2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // GATHER Copy
	2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // GATHER Scale
	3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // GATHER Add
	3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // GATHER Triad
	// Scatter Kernels
	2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // SCATTER Copy
	2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // SCATTER Scale
	3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // SCATTER Add
	3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE  // SCATTER Triad
};

static double   flops[NUM_KERNELS] = {
	// Original Kernels
	(int)0,                // Copy
	1 * STREAM_ARRAY_SIZE, // Scale
	1 * STREAM_ARRAY_SIZE, // Add
	2 * STREAM_ARRAY_SIZE, // Triad
	// Gather Kernels
	(int)0,                // GATHER Copy
	1 * STREAM_ARRAY_SIZE, // GATHER Scale
	1 * STREAM_ARRAY_SIZE, // GATHER Add
	2 * STREAM_ARRAY_SIZE, // GATHER Triad
	// Scatter Kernels
	(int)0,                // SCATTER Copy
	1 * STREAM_ARRAY_SIZE, // SCATTER Scale
	1 * STREAM_ARRAY_SIZE, // SCATTER Add
	2 * STREAM_ARRAY_SIZE, // SCATTER Triad
};

extern void init_idx_array(int *array, int nelems);

extern void print_info1(int BytesPerWord, int numranks, ssize_t array_elements, int k);
extern void print_timer_granularity(int quantum);
extern void print_info2(double t, double t0, double t1, int quantum);
extern void print_memory_usage();

extern void checkSTREAMresults(STREAM_TYPE *AvgErrByRank, int numranks);
extern void computeSTREAMerrors(STREAM_TYPE *aAvgErr, STREAM_TYPE *bAvgErr, STREAM_TYPE *cAvgErr);

double mysecond();

#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(STREAM_TYPE scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(STREAM_TYPE scalar);
#endif

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

static double times[NUM_KERNELS][NTIMES];
static STREAM_TYPE AvgError[NUM_ARRAYS];

int main()
{
    int			quantum, checktick();
    int			BytesPerWord;
    int			i,k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t;
	double		*TimesByRank;
	double		t0,t1,tmin;
	int         rc, numranks, myrank;
	STREAM_TYPE *AvgErrByRank;

/*--------------------------------------------------------------------------------------
    - Setup MPI
--------------------------------------------------------------------------------------*/
    rc = MPI_Init(NULL, NULL);
	t0 = MPI_Wtime();
    if (rc != MPI_SUCCESS) {
       printf("ERROR: MPI Initialization failed with return code %d\n",rc);
       exit(1);
    }
	// if either of these fail there is something really screwed up!
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

/*--------------------------------------------------------------------------------------
    - Set the average errror for each array to 0 as default, since we haven't done
        anything with the arrays yet. AvgErrByPE will be updated using an OpenSHMEM
        collective later on.
--------------------------------------------------------------------------------------*/
    for (int i=0;i<NUM_ARRAYS;i++) {
        AvgError[i] = 0.0;
    }

/*--------------------------------------------------------------------------------------
    - Set the mintime to default value (FLT_MAX) for each kernel, since we haven't executed
        any of the kernels or done any timing yet
--------------------------------------------------------------------------------------*/
    for (int i=0;i<NUM_KERNELS;i++) {
        mintime[i] = FLT_MAX;
    }

/*--------------------------------------------------------------------------------------
    - Initialize the idx arrays on all PEs and populate them with random values ranging
        from 0 - STREAM_ARRAY_SIZE
--------------------------------------------------------------------------------------*/
    srand(time(0));
    init_idx_array(a_idx, STREAM_ARRAY_SIZE/numranks);
    init_idx_array(b_idx, STREAM_ARRAY_SIZE/numranks);
    init_idx_array(c_idx, STREAM_ARRAY_SIZE/numranks);

/*--------------------------------------------------------------------------------------
    - Initialize the idx arrays on all PEs and populate them with random values ranging
        from 0 - STREAM_ARRAY_SIZE
--------------------------------------------------------------------------------------*/
    /* --- NEW FEATURE --- distribute requested storage across MPI ranks --- */
	array_elements = STREAM_ARRAY_SIZE / numranks;		// don't worry about rounding vs truncation
    array_alignment = 64;						// Can be modified -- provides partial support for adjusting relative alignment

	// Dynamically allocate the arrays using "posix_memalign()"
	// NOTE that the OFFSET parameter is not used in this version of the code!
    array_bytes = array_elements * sizeof(STREAM_TYPE);

/*--------------------------------------------------------------------------------------
	- Allocate memory for the STREAM arrays
--------------------------------------------------------------------------------------*/
    k = posix_memalign((void **)&a, array_alignment, array_bytes);
    if (k != 0) {
        printf("Rank %d: Allocation of array a failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&b, array_alignment, array_bytes);
    if (k != 0) {
        printf("Rank %d: Allocation of array b failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }
    k = posix_memalign((void **)&c, array_alignment, array_bytes);
    if (k != 0) {
        printf("Rank %d: Allocation of array c failed, return code is %d\n",myrank,k);
		MPI_Abort(MPI_COMM_WORLD, 2);
        exit(1);
    }

/*--------------------------------------------------------------------------------------
	// Initial informational printouts -- rank 0 handles all the output
--------------------------------------------------------------------------------------*/
	if (myrank == 0) {
        print_info1(BytesPerWord, numranks, array_elements, k);
	}

/*--------------------------------------------------------------------------------------
    // Populate STREAM arrays on all ranks
--------------------------------------------------------------------------------------*/
#pragma omp parallel for
    for (j=0; j<array_elements; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

	// Rank 0 needs to allocate arrays to hold error data and timing data from
	// all ranks for analysis and output.
	// Allocate and instantiate the arrays here -- after the primary arrays
	// have been instantiated -- so there is no possibility of having these
	// auxiliary arrays mess up the NUMA placement of the primary arrays.
	if (myrank == 0) {
		// There are NUM_ARRAYS average error values for each rank (using STREAM_TYPE).
		AvgErrByRank = (double *) malloc(NUM_ARRAYS * sizeof(STREAM_TYPE) * numranks);
		if (AvgErrByRank == NULL) {
			printf("Ooops -- allocation of arrays to collect errors on MPI rank 0 failed\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		memset(AvgErrByRank,0,NUM_ARRAYS*sizeof(STREAM_TYPE)*numranks);

		// There are NUM_KERNELS*NTIMES timing values for each rank (always doubles)
		TimesByRank = (double *) malloc(NUM_KERNELS * NTIMES * sizeof(double) * numranks);
		if (TimesByRank == NULL) {
			printf("Ooops -- allocation of arrays to collect timing data on MPI rank 0 failed\n");
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		memset(TimesByRank,0,NUM_KERNELS*NTIMES*sizeof(double)*numranks);
	}

/*--------------------------------------------------------------------------------------
    // Estimate precision and granularity of timer
--------------------------------------------------------------------------------------*/
	// Simple check for granularity of the timer being used
	if (myrank == 0) {
        print_timer_granularity(quantum);
	}

    /* Get initial timing estimate to compare to timer granularity. */
	/* All ranks need to run this code since it changes the values in array a */
    t = MPI_Wtime();
#pragma omp parallel for
    for (j = 0; j < array_elements; j++)
		a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (MPI_Wtime() - t);

	if (myrank == 0) {
        print_info2(t, t0, t1, quantum);
		print_memory_usage();
	}

// =================================================================================
    		/*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
// =================================================================================

    // This code has more barriers and timing calls than are actually needed, but
    // this should not cause a problem for arrays that are large enough to satisfy
    // the STREAM run rules.
	// MAJOR FIX!!!  Version 1.7 had the start timer for each loop *after* the
	// MPI_Barrier(), when it should have been *before* the MPI_Barrier().
    //

    scalar = SCALAR;
    for (k=0; k<NTIMES; k++)
	{
// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
// ----------------------------------------------
// 				  COPY KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		// t0 = mysecond();
		MPI_Barrier(MPI_COMM_WORLD);		

#ifdef TUNED
        tuned_STREAM_Copy();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			c[j] = a[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		// t1 = mysecond();
		t1 = MPI_Wtime();
		times[0][k] = t1 - t0;

// ----------------------------------------------
// 		 	     SCALE KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			b[j] = scalar*c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[1][k] = t1-t0;

// ----------------------------------------------
// 				 ADD KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			c[j] = a[j]+b[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[2][k] = t1-t0;

// ----------------------------------------------
//				TRIAD KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			a[j] = b[j]+scalar*c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[3][k] = t1-t0;

// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
// ----------------------------------------------
// 				GATHER COPY KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Copy_Gather();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			c[j] = a[a_idx[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[4][k] = t1 - t0;

// ----------------------------------------------
// 				GATHER SCALE KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Scale_Gather();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			b[j] = scalar * c[c_idx[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[5][k] = t1 - t0;

// ----------------------------------------------
// 				GATHER ADD KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Add_Gather();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			c[j] = a[a_idx[j]] + b[b_idx[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[6][k] = t1 - t0;

// ----------------------------------------------
// 			   GATHER TRIAD KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Triad_Gather();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			a[j] = b[b_idx[j]] + scalar * c[c_idx[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[7][k] = t1 - t0;

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
// ----------------------------------------------
// 				SCATTER COPY KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Copy_Scatter();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			c[c_idx[j]] = a[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[8][k] = t1 - t0;

// ----------------------------------------------
// 				SCATTER SCALE KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Scale_Scatter();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			b[b_idx[j]] = scalar * c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[9][k] = t1 - t0;

// ----------------------------------------------
// 				SCATTER ADD KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Add_Scatter();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			c[c_idx[j]] = a[j] + b[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[10][k] = t1 - t0;

// ----------------------------------------------
// 				SCATTER TRIAD KERNEL
// ----------------------------------------------
		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
		tuned_STREAM_Triad_Scatter();
#else
#pragma omp parallel for
		for (j=0; j<array_elements; j++)
			a[a_idx[j]] = b[j] + scalar * c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[11][k] = t1 - t0;
	}

	t0 = MPI_Wtime();

    /*	--- SUMMARY --- */

	// Because of the MPI_Barrier() calls, the timings from any thread are equally valid.
    // The best estimate of the maximum performance is the minimum of the "outside the barrier"
    // timings across all the MPI ranks.

/*--------------------------------------------------------------------------------------
	// Gather all timing data to MPI rank 0
--------------------------------------------------------------------------------------*/
	MPI_Gather(times, NUM_KERNELS*NTIMES, MPI_DOUBLE, TimesByRank, NUM_KERNELS*NTIMES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*--------------------------------------------------------------------------------------
	Rank 0 processes all timing data
--------------------------------------------------------------------------------------*/
	if (myrank == 0) {
		// for each iteration and each kernel, collect the minimum time across all MPI ranks
		// and overwrite the rank 0 "times" variable with the minimum so the original post-
		// processing code can still be used.
		for (k=0; k<NTIMES; k++) {
			for (j=0; j<NUM_KERNELS; j++) {
				tmin = 1.0e36;
				for (i=0; i<numranks; i++) {
					// printf("DEBUG: Timing: iter %d, kernel %lu, rank %d, tmin %f, TbyRank %f\n",k,j,i,tmin,TimesByRank[NUM_KERNELS*NTIMES*i+j*NTIMES+k]);
					tmin = MIN(tmin, TimesByRank[NUM_KERNELS*NTIMES*i+j*NTIMES+k]);
				}
				// printf("DEBUG: Final Timing: iter %d, kernel %lu, final tmin %f\n",k,j,tmin);
				times[j][k] = tmin;
			}
		}

/*--------------------------------------------------------------------------------------
	Back to the original code, but now using the minimum global timing across all ranks
--------------------------------------------------------------------------------------*/
		for (k=1; k<NTIMES; k++) { /* note -- skip first iteration */
            for (j=0; j<NUM_KERNELS; j++) {
                avgtime[j] = avgtime[j] + times[j][k];
                mintime[j] = MIN(mintime[j], times[j][k]);
                maxtime[j] = MAX(maxtime[j], times[j][k]);
            }
		}

		// note that "bytes[j]" is the aggregate array size, so no "numranks" is needed here
		printf("Function\tBest Rate MB/s\t      FLOP/s\t   Avg time\t   Min time\t   Max time\n");
		for (j=0; j<NUM_KERNELS; j++) {
			avgtime[j] = avgtime[j]/(double)(NTIMES-1);
			
			if (flops[j] == 0) {
				printf("%s%12.1f\t\t%s\t%11.6f\t%11.6f\t%11.6f\n",
					label[j],                           // Kernel
					1.0E-06 * bytes[j]/mintime[j],      // MB/s
					"-",      // FLOP/s
					avgtime[j],                         // Avg Time
					mintime[j],                         // Min Time
					maxtime[j]);                        // Max time
			}
			else {
				printf("%s%12.1f\t%12.1f\t%11.6f\t%11.6f\t%11.6f\n",
					label[j],                           // Kernel
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
	// Validate the results
--------------------------------------------------------------------------------------*/
#ifdef INJECTERROR
	a[11] = 100.0 * a[11];
#endif
	/* --- Collect the Average Errors for Each Array on Rank 0 --- */
	computeSTREAMerrors(&AvgError[0], &AvgError[1], &AvgError[2]);
	MPI_Gather(AvgError, NUM_ARRAYS, MPI_DOUBLE, AvgErrByRank, NUM_ARRAYS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* -- Combined averaged errors and report on Rank 0 only --- */
	if (myrank == 0) {
#ifdef VERBOSE
		for (k=0; k<numranks; k++) {
			printf("VERBOSE: rank %d, AvgErrors %e %e %e\n",k,AvgErrByRank[3*k+0],
				AvgErrByRank[3*k+1],AvgErrByRank[3*k+2]);
		}
#endif
		checkSTREAMresults(AvgErrByRank,numranks);
		printf(HLINE);
	}
//------------------------------------------------------------------------------------

#ifdef VERBOSE
	if (myrank == 0) {
		t1 = MPI_Wtime();
		printf("VERBOSE: total shutdown time for rank %d = %f seconds\n",myrank,t1-t0);
	}
#endif

	free(a);
	free(b);
	free(c);
	if (myrank == 0) {
		free(TimesByRank);
	}

    MPI_Finalize();
	return(0);
}




double mysecond()
{
        struct timeval tp;
        // struct timezone tzp;
        int i;

        i = gettimeofday(&tp, NULL);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


# define	M	20
int
checktick() {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = MPI_Wtime();
	while( ((t2=MPI_Wtime()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
		Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
		minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
}


/*--------------------------------------------------------------------------------------
 - Initializes provided array with random indices within data array
    bounds. Forces a one-to-one mapping from available data array indices
    to utilized indices in index array. This simplifies the scatter kernel
    verification process and precludes the need for atomic operations.
--------------------------------------------------------------------------------------*/
void init_idx_array(int *array, int nelems) {
	int i, success, idx;

	// Array to track used indices
	char* flags = (char*) malloc(sizeof(char)*nelems);
	for(i = 0; i < nelems; i++){
		flags[i] = 0;
	}

	// Iterate and fill each element of the idx array
	for (i = 0; i < nelems; i++) {
		success = 0;
		while(success == 0){
			idx = ((int) rand()) % nelems;
			if(flags[idx] == 0){
				array[i] = idx;
				flags[idx] = -1;
				success = 1;
			}
		}
	}
	free(flags);
}


//========================================================================================
// 				VALIDATION PIECE
//========================================================================================
#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
void computeSTREAMerrors(STREAM_TYPE *aAvgErr, STREAM_TYPE *bAvgErr, STREAM_TYPE *cAvgErr)
{
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aSumErr,bSumErr,cSumErr;
	ssize_t	j;
	int	k;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

    /* now execute timing loop */
	scalar = SCALAR;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;

            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;

            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (j=0; j<array_elements; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}
	*aAvgErr = aSumErr / (STREAM_TYPE) array_elements;
	*bAvgErr = bSumErr / (STREAM_TYPE) array_elements;
	*cAvgErr = cSumErr / (STREAM_TYPE) array_elements;
}



void checkSTREAMresults (STREAM_TYPE *AvgErrByRank, int numranks)
{
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aSumErr,bSumErr,cSumErr;
	STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
	double epsilon;
	ssize_t	j;
	int	k,ierr,err;

	// Repeat the computation of aj, bj, cj because I am lazy
    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = SCALAR;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;

            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;

            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

	// Compute the average of the average errors contributed by each MPI rank
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (k=0; k<numranks; k++) {
		aSumErr += AvgErrByRank[3*k + 0];
		bSumErr += AvgErrByRank[3*k + 1];
		cSumErr += AvgErrByRank[3*k + 2];
	}
	aAvgErr = aSumErr / (STREAM_TYPE) numranks;
	bAvgErr = bSumErr / (STREAM_TYPE) numranks;
	cAvgErr = cSumErr / (STREAM_TYPE) numranks;

	if (sizeof(STREAM_TYPE) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(STREAM_TYPE) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
		epsilon = 1.e-6;
	}

	err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (j=0; j<array_elements; j++) {
			if (abs(a[j]/aj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,aj,a[j],abs((aj-a[j])/aAvgErr));
				}
#endif
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
	}
	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<array_elements; j++) {
			if (abs(b[j]/bj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,bj,b[j],abs((bj-b[j])/bAvgErr));
				}
#endif
			}
		}
		printf("     For array b[], %d errors were found.\n",ierr);
	}
	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<array_elements; j++) {
			if (abs(c[j]/cj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,cj,c[j],abs((cj-c[j])/cAvgErr));
				}
#endif
			}
		}
		printf("     For array c[], %d errors were found.\n",ierr);
	}
	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

void print_info1(int BytesPerWord, int numranks, ssize_t array_elements, int k) {
    printf(HLINE);
		printf("RaiderSTREAM\n");
		printf(HLINE);
		BytesPerWord = sizeof(STREAM_TYPE);
		printf("This system uses %d bytes per array element.\n", BytesPerWord);

		printf(HLINE);
#ifdef N
		printf("*****  WARNING: ******\n");
		printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
		printf("      This version of the code uses the preprocesor variable STREAM_ARRAY_SIZE to control the array size\n");
		printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",(unsigned long long) STREAM_ARRAY_SIZE);
		printf("*****  WARNING: ******\n");
#endif
		if (OFFSET != 0) {
			printf("*****  WARNING: ******\n");
			printf("   This version ignores the OFFSET parameter.\n");
			printf("*****  WARNING: ******\n");
		}

		printf("Total Aggregate Array size = %llu (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE);
		printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n",
			BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
			BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
		printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
			(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
			(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
		printf("Data is distributed across %d MPI ranks\n",numranks);
		printf("   Array size per MPI rank = %llu (elements)\n" , (unsigned long long) array_elements);
		printf("   Memory per array per MPI rank = %.1f MiB (= %.1f GiB).\n",
			BytesPerWord * ( (double) array_elements / 1024.0/1024.0),
			BytesPerWord * ( (double) array_elements / 1024.0/1024.0/1024.0));
		printf("   Total memory per MPI rank = %.1f MiB (= %.1f GiB).\n",
			(3.0 * BytesPerWord) * ( (double) array_elements / 1024.0/1024.),
			(3.0 * BytesPerWord) * ( (double) array_elements / 1024.0/1024./1024.));

		printf(HLINE);
		printf("Each kernel will be executed %d times.\n", NTIMES);
		printf(" The *best* time for each kernel (excluding the first iteration)\n");
		printf(" will be used to compute the reported bandwidth.\n");
		printf("The SCALAR value used for this run is %f\n",SCALAR);

#ifdef _OPENMP
		printf(HLINE);
#pragma omp parallel
		{
#pragma omp master
		{
			k = omp_get_num_threads();
			printf ("Number of Threads requested for each MPI rank = %i\n",k);
			}
		}
#endif

#ifdef _OPENMP
		k = 0;
#pragma omp parallel
#pragma omp atomic
			k++;
		printf ("Number of Threads counted for rank 0 = %i\n",k);
#endif
}

void print_timer_granularity(int quantum) {
    printf(HLINE);

    if  ( (quantum = checktick()) >= 1)
    printf("Your timer granularity/precision appears to be "
        "%d microseconds.\n", quantum);
    else {
    printf("Your timer granularity appears to be "
        "less than one microsecond.\n");
    quantum = 1;
    }
}

void print_info2(double t, double t0, double t1, int quantum) {
    	printf("Each test below will take on the order"
		" of %d microseconds.\n", (int) t  );
		// printf("   (= %d timer ticks)\n", (int) (t/quantum) );
		printf("   (= %d timer ticks)\n", (int) (t) );
		printf("Increase the size of the arrays if this shows that\n");
		printf("you are not getting at least 20 timer ticks per test.\n");

		printf(HLINE);

		printf("WARNING -- The above is only a rough guideline.\n");
		printf("For best results, please be sure you know the\n");
		printf("precision of your system timer.\n");
		printf(HLINE);
#ifdef VERBOSE
		t1 = MPI_Wtime();
		printf("VERBOSE: total setup time for rank 0 = %f seconds\n", t1 - t0);
		printf(HLINE);
#endif
}


void print_memory_usage() { // TODO: FACTOR IN NUMBER OF CORES - THIS IS ONLY FOR SINGLE CORE
	unsigned long totalMemory = \
		(sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) + 	// a[]
		(sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) + 	// b[]
		(sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) + 	// c[]
		(sizeof(int) * (STREAM_ARRAY_SIZE)) + 			// a_idx[]
		(sizeof(int) * (STREAM_ARRAY_SIZE)) + 			// b_idx[]
		(sizeof(int) * (STREAM_ARRAY_SIZE)) + 			// c_idx[]
		(sizeof(double) * NUM_KERNELS) + 				// avgtime[]
		(sizeof(double) * NUM_KERNELS) + 				// maxtime[]
		(sizeof(double) * NUM_KERNELS) + 				// mintime[]
		(sizeof(char) * NUM_KERNELS) +					// label[]
		(sizeof(double) * NUM_KERNELS);					// bytes[]

#ifdef VERBOSE // if -DVERBOSE enabled break down memory usage by each array
	printf("---------------------------------\n");
	printf("  VERBOSE Memory Breakdown\n");
	printf("---------------------------------\n");
	printf("a[]:\t\t%.2f MB\n", (sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) / 1024.0 / 1024.0);
	printf("b[]:\t\t%.2f MB\n", (sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) / 1024.0 / 1024.0);
	printf("c[]:\t\t%.2f MB\n", (sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) / 1024.0 / 1024.0);
	printf("a_idx[]:\t%.2f MB\n", (sizeof(int) * (STREAM_ARRAY_SIZE)) / 1024.0 / 1024.0);
	printf("b_idx[]:\t%.2f MB\n", (sizeof(int) * (STREAM_ARRAY_SIZE)) / 1024.0 / 1024.0);
	printf("c_idx[]:\t%.2f MB\n", (sizeof(int) * (STREAM_ARRAY_SIZE)) / 1024.0 / 1024.0);
	printf("avgtime[]:\t%lu B\n", (sizeof(double) * NUM_KERNELS));
	printf("maxtime[]:\t%lu B\n", (sizeof(double) * NUM_KERNELS));
	printf("mintime[]:\t%lu B\n", (sizeof(double) * NUM_KERNELS));
	printf("label[]:\t%lu B\n", (sizeof(char) * NUM_KERNELS));
	printf("bytes[]:\t%lu B\n", (sizeof(double) * NUM_KERNELS));
	printf("---------------------------------\n");
	printf("Total Memory Allocated: %.2f MB\n", totalMemory / 1024.0 / 1024.0);
	printf("---------------------------------\n");
#else
	printf("Totaly Memory Allocated: %.2f MB\n", totalMemory / 1024.0 / 1024.0);
#endif
	printf(HLINE);
}






/* stubs for "tuned" versions of the kernels */
#ifdef TUNED
// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
void tuned_STREAM_Copy()
{
	ssize_t j;
#pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++)
        c[j] = a[j];
}

void tuned_STREAM_Scale(STREAM_TYPE scalar)
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    b[j] = scalar*c[j];
}

void tuned_STREAM_Add()
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    a[j] = b[j]+scalar*c[j];
}
// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Gather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		c[j] = a[a_idx[j]];
}

void tuned_STREAM_Scale_Gather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		b[j] = scalar * c[c_idx[j]];
}

void tuned_STREAM_Add_Gather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		c[j] = a[a_idx[j]] + b[b_idx[j]];
}

void tuned_STREAM_Triad_Gather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		a[j] = b[b_idx[j]] + scalar * c[c_idx[j]];
}

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Scatter() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		c[c_idx[j]] = a[j];
}

void tuned_STREAM_Scale_Scatter(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		b[b_idx[j]] = scalar * c[j];
}

void tuned_STREAM_Add_Scatter() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		c[a_idx[j]] = a[j] + b[j];
}

void tuned_STREAM_Triad_Scatter(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		a[a_idx[j]] = b[j] + scalar * c[j];
}
/* end of stubs for the "tuned" versions of the kernels */
#endif

