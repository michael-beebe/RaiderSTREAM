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

#include "stream_mpi_output.h"
#include "stream_mpi_tuned.h"
#include "stream_mpi_validation.h"

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
static double times[NUM_KERNELS][NTIMES];
static STREAM_TYPE AvgError[NUM_ARRAYS];
static int is_validated[NUM_KERNELS] = {0};

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

void init_arrays(ssize_t stream_array_size) {
	#pragma omp parallel for
    for (ssize_t j = 0; j < stream_array_size; j++) {
		a[j] = 2.0; // 1 or 2? since we are changing the validation we could discuss
		b[j] = 2.0;
		c[j] = 0.0;
    }
}

void stream_copy(double times[NUM_KERNELS][NTIMES], int k) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Copy();
#else
#pragma omp parallel for
		for ( j = 0; j < array_elements; j++)
			c[j] = a[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[COPY][k] = t1 - t0;
}

void stream_scale(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			b[j] = scalar * c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[SCALE][k] = t1 - t0;
}

void stream_sum(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			c[j] = a[j] + b[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[SUM][k] = t1 - t0;
}

void stream_triad(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			a[j] = b[j] + scalar * c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[TRIAD][k] = t1 - t0;
}


void gather_copy(double times[NUM_KERNELS][NTIMES], int k) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Copy_Gather();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			c[j] = a[IDX1[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[GATHER_COPY][k] = t1 - t0;
}

void gather_scale(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Scale_Gather();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			b[j] = scalar * c[IDX2[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[GATHER_SCALE][k] = t1 - t0;
}

void gather_sum(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Add_Gather();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			c[j] = a[IDX1[j]] + b[IDX2[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[GATHER_SUM][k] = t1 - t0;
}

void gather_triad(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			a[j] = b[IDX1[j]] + scalar * c[IDX2[j]];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[GATHER_TRIAD][k] = t1 - t0;
}


void scatter_copy(double times[NUM_KERNELS][NTIMES], int k) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Copy_Scatter();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			c[IDX1[j]] = a[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[SCATTER_COPY][k] = t1 - t0;
}

void scatter_scale(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Scale_Scatter();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			b[IDX2[j]] = scalar * c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[SCATTER_SCALE][k] = t1 - t0;
}

void scatter_sum(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef TUNED
        tuned_STREAM_Add_Scatter();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			c[IDX1[j]] = a[j] + b[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[SCATTER_SUM][k] = t1 - t0;
}

void scatter_triad(double times[NUM_KERNELS][NTIMES], int k, double scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Triad_Scatter();
#else
#pragma omp parallel for
		for (j = 0; j < array_elements; j++)
			a[IDX2[j]] = b[j] + scalar * c[j];
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		times[SCATTER_TRIAD][k] = t1 - t0;
}

void central_copy(double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
    	tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j = 0; j < array_elements; j++)
		c[0] = a[0];
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[CENTRAL_COPY][k] = t1 - t0;
}

void central_scale(double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < array_elements; j++)
	    b[0] = scalar * c[0];
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[CENTRAL_SCALE][k] = t1 - t0;
}

void central_sum(double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j = 0; j < array_elements; j++)
	    c[0] = a[0] + b[0];
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[CENTRAL_SUM][k] = t1 - t0;
}

void central_triad(double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < array_elements; j++)
	    a[0] = b[0] + scalar * c[0];
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	times[CENTRAL_TRIAD][k] = t1 - t0;
}


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
		// Central Kernels
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // CENTRAL Copy
		2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // CENTRAL Scale
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // CENTRAL Add
		3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE, // CENTRAL Triad
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
		// Central Kernels
		(int)0,                // CENTRAL Copy
		1 * STREAM_ARRAY_SIZE, // CENTRAL Scale
		1 * STREAM_ARRAY_SIZE, // CENTRAL Add
		2 * STREAM_ARRAY_SIZE, // CENTRAL Triad
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
	IDX1 = malloc(array_elements * sizeof(ssize_t));
	IDX2 = malloc(array_elements * sizeof(ssize_t));

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
// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				  COPY KERNEL
	// ----------------------------------------------
		stream_copy(times, k);
	// ----------------------------------------------
	// 		 	     SCALE KERNEL
	// ----------------------------------------------
		stream_scale(times, k, scalar);
	// ----------------------------------------------
	// 				 ADD KERNEL
	// ----------------------------------------------
		stream_sum(times, k, scalar);
	// ----------------------------------------------
	//				TRIAD KERNEL
	// ----------------------------------------------
		stream_triad(times, k, scalar);
	}

	// ----------------------------------------------
	//				VALIDATION
	// ----------------------------------------------
	stream_validation(array_elements, scalar, is_validated, a, b, c, myrank, numranks);
	
	MPI_Barrier(MPI_COMM_WORLD);
// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
	init_arrays(array_elements);
	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				GATHER COPY KERNEL
	// ----------------------------------------------
		gather_copy(times, k);
	// ----------------------------------------------
	// 				GATHER SCALE KERNEL
	// ----------------------------------------------
		gather_scale(times, k, scalar);
	// ----------------------------------------------
	// 				GATHER ADD KERNEL
	// ----------------------------------------------
		gather_sum(times, k, scalar);
	// ----------------------------------------------
	// 				GATHER TRIAD KERNEL
	// ----------------------------------------------
		gather_triad(times, k, scalar);
	}

	gather_validation(array_elements, scalar, is_validated, a, b, c, myrank, numranks);
	
	MPI_Barrier(MPI_COMM_WORLD);
// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
	init_arrays(array_elements);
	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				SCATTER COPY KERNEL
	// ----------------------------------------------
		scatter_copy(times, k);
	// ----------------------------------------------
	// 				SCATTER SCALE KERNEL
	// ----------------------------------------------
		scatter_scale(times, k, scalar);
	// ----------------------------------------------
	// 				SCATTER ADD KERNEL
	// ----------------------------------------------
		scatter_sum(times, k, scalar);
	// ----------------------------------------------
	// 				SCATTER TRIAD KERNEL
	// ----------------------------------------------
		scatter_triad(times, k, scalar);
	}

	scatter_validation(array_elements, scalar, is_validated, a, b, c, myrank, numranks);
	
	MPI_Barrier(MPI_COMM_WORLD);

// =================================================================================
//						CENTRAL VERSIONS OF THE KERNELS
// =================================================================================
	init_arrays(array_elements);

	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				CENTRAL COPY KERNEL
	// ----------------------------------------------
		central_copy(times, k, scalar);
	// ----------------------------------------------
	// 				CENTRAL SCALE KERNEL
	// ----------------------------------------------
		central_scale(times, k, scalar);
	// ----------------------------------------------
	// 				CENTRAL ADD KERNEL
	// ----------------------------------------------
		central_sum(times, k, scalar);
	// ----------------------------------------------
	// 				CENTRAL TRIAD KERNEL
	// ----------------------------------------------
		central_triad(times, k, scalar);
	}

	central_validation(array_elements, scalar, is_validated, a, b, c, myrank, numranks);

	MPI_Barrier(MPI_COMM_WORLD);

#ifdef VERBOSE
	t0 = MPI_Wtime();
#endif

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

/*--------------------------------------------------------------------------------------
	Back to the original code, but now using the minimum global timing across all ranks
--------------------------------------------------------------------------------------*/
		for (k = 1; k < NTIMES; k++) { /* note -- skip first iteration */
            for (j = 0; j < NUM_KERNELS; j++) {
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
	if (myrank == 0) {
#ifdef VERBOSE
		for (k=0; k<numranks; k++) {
			printf("VERBOSE: rank %d, AvgErrors %e %e %e\n",k,AvgErrByRank[3*k+0],
				AvgErrByRank[3*k+1],AvgErrByRank[3*k+2]);
		}
#endif
		checkSTREAMresults(is_validated);
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