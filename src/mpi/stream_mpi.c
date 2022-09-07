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

static ssize_t *IDX1;
static ssize_t *IDX2;

ssize_t		array_elements, array_bytes, array_alignment;

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

extern void init_random_idx_array(ssize_t *array, ssize_t nelems);
extern void init_read_idx_array(ssize_t *array, ssize_t nelems, char *filename);

extern void print_info1(int BytesPerWord, int numranks, ssize_t array_elements, int k, ssize_t STREAM_ARRAY_SIZE);
extern void print_timer_granularity(int quantum);
extern void print_info2(double t, double t0, double t1, int quantum);
extern void print_memory_usage(int numranks, ssize_t STREAM_ARRAY_SIZE);

extern void checkSTREAMresults(STREAM_TYPE *AvgErrByRank, int numranks);
extern void computeSTREAMerrors(STREAM_TYPE *aAvgErr, STREAM_TYPE *bAvgErr, STREAM_TYPE *cAvgErr);

extern void parse_opts(int argc, char **argv, ssize_t *STREAM_ARRAY_SIZE);

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

int main(int argc, char *argv[])
{
	ssize_t STREAM_ARRAY_SIZE = 10000000; // Default STREAM_ARRAY_SIZE is 10000000

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

	parse_opts(argc, argv, &STREAM_ARRAY_SIZE);

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

/* --- NEW FEATURE --- distribute requested storage across MPI ranks --- */
	array_elements = STREAM_ARRAY_SIZE / numranks;		// don't worry about rounding vs truncation
	array_alignment = 64;						// Can be modified -- provides partial support for adjusting relative alignment

// Dynamically allocate the arrays using "posix_memalign()"
// NOTE that the OFFSET parameter is not used in this version of the code!
	array_bytes = array_elements * sizeof(STREAM_TYPE);

/*--------------------------------------------------------------------------------------
- Initialize array for storing the number of bytes that needs to be counted for
  each benchmark kernel
--------------------------------------------------------------------------------------*/
	double	bytes[NUM_KERNELS] = {
		// Original Kernels
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Copy
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Scale
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Add
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // Triad
		// Gather Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER Scale
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // GATHER Triad
		// Scatter Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER Scale
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * STREAM_ARRAY_SIZE), // SCATTER Triad
	};

	double   flops[NUM_KERNELS] = {
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
- Initialize idx arrays (which will be used by gather/scatter kernels)
--------------------------------------------------------------------------------------*/
	IDX1 = malloc(array_elements * sizeof IDX1);
	IDX2 = malloc(array_elements * sizeof IDX2);

/*--------------------------------------------------------------------------------------
    - Initialize the idx arrays on all PEs
	- Use the input .txt files to populate each array if the -DCUSTOM flag is enabled
	- If -DCUSTOM is not enabled, populate the IDX arrays with random values
--------------------------------------------------------------------------------------*/
    #ifdef CUSTOM
    	init_read_idx_array(IDX1, array_elements, "IDX1.txt");
    	init_read_idx_array(IDX2, array_elements, "IDX2.txt");
    #else
        srand(time(0));
        init_random_idx_array(IDX1, array_elements);
        init_random_idx_array(IDX2, array_elements);
    #endif
	
/*--------------------------------------------------------------------------------------
	// Initial informational printouts -- rank 0 handles all the output
--------------------------------------------------------------------------------------*/
	if (myrank == 0) {
        print_info1(BytesPerWord, numranks, array_elements, k, STREAM_ARRAY_SIZE);
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
		print_memory_usage(numranks, STREAM_ARRAY_SIZE);
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
			c[j] = a[IDX1[j]];
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
			b[j] = scalar * c[IDX2[j]];
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
			c[j] = a[IDX1[j]] + b[IDX2[j]];
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
			a[j] = b[IDX1[j]] + scalar * c[IDX2[j]];
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
			c[IDX1[j]] = a[j];
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
			b[IDX2[j]] = scalar * c[j];
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
			c[IDX1[j]] = a[j] + b[j];
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
			a[IDX2[j]] = b[j] + scalar * c[j];
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
		printf("Function\tBest Rate MB/s      Best FLOP/s\t   Avg time\t   Min time\t   Max time\n");
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
void init_random_idx_array(ssize_t *array, ssize_t nelems) {
	ssize_t i, idx;
	int success = 0;

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

/*--------------------------------------------------------------------------------------
 - Initializes the IDX arrays with the contents of IDX1.txt and IDX2.txt, respectively
--------------------------------------------------------------------------------------*/
void init_read_idx_array(ssize_t *array, ssize_t nelems, char *filename) {
    FILE *file;
    file = fopen(filename, "r");
    if (!file) {
        perror(filename);
        exit(1);
    }

    for (ssize_t i=0; i < nelems; i++) {
        fscanf(file, "%zd", &array[i]);
    }

    fclose(file);
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

//========================================================================================

void print_info1(int BytesPerWord, int numranks, ssize_t array_elements, int k, ssize_t STREAM_ARRAY_SIZE) {
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


void print_memory_usage(int numranks, ssize_t STREAM_ARRAY_SIZE) {
	unsigned long totalMemory = \
		((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) + 	// a[]
		((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) + 	// b[]
		((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) + 	// c[]
		((sizeof(int) * (STREAM_ARRAY_SIZE)) * numranks) + 			// a_idx[]
		((sizeof(int) * (STREAM_ARRAY_SIZE)) * numranks) + 			// b_idx[]
		((sizeof(int) * (STREAM_ARRAY_SIZE)) * numranks) + 			// c_idx[]
		((sizeof(double) * NUM_KERNELS) * numranks) + 				// avgtime[]
		((sizeof(double) * NUM_KERNELS) * numranks) + 				// maxtime[]
		((sizeof(double) * NUM_KERNELS) * numranks) + 				// mintime[]
		((sizeof(char) * NUM_KERNELS) * numranks) +					// label[]
		((sizeof(double) * NUM_KERNELS) * numranks);				// bytes[]

#ifdef VERBOSE // if -DVERBOSE enabled break down memory usage by each array
	printf("-----------------------------------------\n");
	printf("     VERBOSE Memory Breakdown\n");
	printf("-----------------------------------------\n");
	printf("a[]:\t\t%.2f MB\n", ((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("b[]:\t\t%.2f MB\n", ((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("c[]:\t\t%.2f MB\n", ((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("a_idx[]:\t%.2f MB\n", ((sizeof(int) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("b_idx[]:\t%.2f MB\n", ((sizeof(int) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("c_idx[]:\t%.2f MB\n", ((sizeof(int) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("avgtime[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("maxtime[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("mintime[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("label[]:\t%lu B\n", ((sizeof(char) * NUM_KERNELS) * numranks));
	printf("bytes[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("-----------------------------------------\n");
	printf("Total Memory Allocated Across All Ranks: %.2f MB\n", totalMemory / 1024.0 / 1024.0);
	printf("-----------------------------------------\n");
#else
	printf("Totaly Memory Allocated Across All Ranks: %.2f MB\n", totalMemory / 1024.0 / 1024.0);
#endif
	printf(HLINE);
}

void parse_opts(int argc, char **argv, ssize_t *STREAM_ARRAY_SIZE) {
    int option;
    while( (option = getopt(argc, argv, "n:t:h")) != -1 ) {
        switch (option) {
            case 'n':
                *STREAM_ARRAY_SIZE = atoi(optarg);
                break;
            case 'h':
                printf("Usage: -n <STREAM_ARRAY_SIZE>\n");
                exit(2);
			
        }
    }
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
