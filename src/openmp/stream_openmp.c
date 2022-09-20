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

#include "stream_openmp_output.h"
#include "stream_openmp_tuned.h"
#include "stream_openmp_validation.h"

// /*--------------------------------------------------------------------------------------
// - Initialize the STREAM arrays used in the kernels
// - Some compilers require an extra keyword to recognize the "restrict" qualifier.
// --------------------------------------------------------------------------------------*/
STREAM_TYPE * restrict a;
STREAM_TYPE * restrict b;
STREAM_TYPE * restrict c;

/*--------------------------------------------------------------------------------------
- Initialize IDX arrays (which will be used by gather/scatter kernels)
--------------------------------------------------------------------------------------*/
static ssize_t *IDX1;
static ssize_t *IDX2;

/*--------------------------------------------------------------------------------------
- Initialize arrays to store avgtime, maxime, and mintime metrics for each kernel.
- The default values are 0 for avgtime and maxtime.
- each mintime[] value needs to be set to FLT_MAX via a for loop inside main()
--------------------------------------------------------------------------------------*/
static double avgtime[NUM_KERNELS] = {0};
static double maxtime[NUM_KERNELS] = {0};
static double mintime[NUM_KERNELS];
static int is_validated[NUM_KERNELS] = {0};

void init_arrays(ssize_t stream_array_size) {
	ssize_t j;
	
	#pragma omp parallel for private (j)
    for (j = 0; j < stream_array_size; j++) {
		a[j] = 2.0; // 1 or 2? since we are changing the validation we could discuss
		b[j] = 2.0;
		c[j] = 0.0;
    }
}


void stream_copy(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
    	tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		c[j] = a[j];
#endif
	t1 = mysecond();
	times[COPY][k] = t1 - t0;
}

void stream_scale(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
	    b[j] = scalar * c[j];
#endif
	t1 = mysecond();
	times[SCALE][k] = t1 - t0;
}

void stream_sum(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
	    c[j] = a[j] + b[j];
#endif
	t1 = mysecond();
	times[SUM][k] = t1 - t0;
}

void stream_triad(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
	    a[j] = b[j] + scalar * c[j];
#endif
	t1 = mysecond();
	times[TRIAD][k] = t1 - t0;
}


void gather_copy(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
	tuned_STREAM_Copy_Gather(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		c[j] = a[IDX1[j]];
#endif
	t1 = mysecond();
	times[GATHER_COPY][k] = t1 - t0;
}

void gather_scale(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Scale_Gather(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		b[j] = scalar * c[IDX2[j]];
#endif
	t1 = mysecond();
	times[GATHER_SCALE][k] = t1 - t0;	
}

void gather_sum(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) { // sum or add ?
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Add_Gather();
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		c[j] = a[IDX1[j]] + b[IDX2[j]];
#endif
	t1 = mysecond();
	times[GATHER_SUM][k] = t1 - t0;
}

void gather_triad(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Triad_Gather(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		a[j] = b[IDX1[j]] + scalar * c[IDX2[j]];
#endif
	t1 = mysecond();
	times[GATHER_TRIAD][k] = t1 - t0;
}

void scatter_copy(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
	tuned_STREAM_Copy_Gather(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		c[IDX1[j]] = a[j];
#endif
	t1 = mysecond();
	times[SCATTER_COPY][k] = t1 - t0;
}

void scatter_scale(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Scale_Gather(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		b[IDX2[j]] = scalar * c[j];
#endif
	t1 = mysecond();
	times[SCATTER_SCALE][k] = t1 - t0;	
}

void scatter_sum(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) { // sum or add ?
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Add_Gather();
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		c[IDX1[j]] = a[j] + b[j];
#endif
	t1 = mysecond();
	times[SCATTER_SUM][k] = t1 - t0;
}

void scatter_triad(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Triad_Gather(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		a[IDX2[j]] = b[j] + scalar * c[j];
#endif
	t1 = mysecond();
	times[SCATTER_TRIAD][k] = t1 - t0;
}

void central_copy(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
    	tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
		c[0] = a[0];
#endif
	t1 = mysecond();
	times[CENTRAL_COPY][k] = t1 - t0;
}

void central_scale(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
	    b[0] = scalar * c[0];
#endif
	t1 = mysecond();
	times[CENTRAL_SCALE][k] = t1 - t0;
}

void central_sum(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;

	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
	    c[0] = a[0] + b[0];
#endif
	t1 = mysecond();
	times[CENTRAL_SUM][k] = t1 - t0;
}

void central_triad(ssize_t stream_array_size, double times[NUM_KERNELS][NTIMES], int k, STREAM_TYPE scalar) {
	double t0, t1;
	ssize_t j;
	
	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j = 0; j < stream_array_size; j++)
	    a[0] = b[0] + scalar * c[0];
#endif
	t1 = mysecond();
	times[CENTRAL_TRIAD][k] = t1 - t0;
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

	parse_opts(argc, argv, &stream_array_size);

	a = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * stream_array_size+OFFSET);
	b = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * stream_array_size+OFFSET);
	c = (STREAM_TYPE *) malloc(sizeof(STREAM_TYPE) * stream_array_size+OFFSET);


	IDX1 = (ssize_t *) malloc(sizeof(ssize_t) * stream_array_size+OFFSET);
	IDX2 = (ssize_t *) malloc(sizeof(ssize_t) * stream_array_size+OFFSET);


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
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // SCATTER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(ssize_t))) * stream_array_size), // SCATTER Triad
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
#else
    srand(time(0));
    init_random_idx_array(IDX1, stream_array_size);
    init_random_idx_array(IDX2, stream_array_size);
#endif

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
// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				  COPY KERNEL
	// ----------------------------------------------
		stream_copy(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 		 	     SCALE KERNEL
	// ----------------------------------------------
		stream_scale(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				 ADD KERNEL
	// ----------------------------------------------
		stream_sum(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	//				TRIAD KERNEL
	// ----------------------------------------------
		stream_triad(stream_array_size, times, k, scalar);
	}
	
	// ----------------------------------------------
	//				VALIDATION
	// ----------------------------------------------
	stream_validation(stream_array_size, scalar, is_validated, a, b, c);

// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
	init_arrays(stream_array_size);
	
	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				GATHER COPY KERNEL
	// ----------------------------------------------
		gather_copy(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				GATHER SCALE KERNEL
	// ----------------------------------------------
		gather_scale(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				GATHER ADD KERNEL
	// ----------------------------------------------
		gather_sum(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 			   GATHER TRIAD KERNEL
	// ----------------------------------------------
		gather_triad(stream_array_size, times, k, scalar);
	}
	
	// ----------------------------------------------
	// 			   GATHER VALIDATION
	// ----------------------------------------------
	gather_validation(stream_array_size, scalar, is_validated, a, b, c);

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
	init_arrays(stream_array_size);

	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				SCATTER COPY KERNEL
	// ----------------------------------------------
		scatter_copy(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				SCATTER SCALE KERNEL
	// ----------------------------------------------
		scatter_scale(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				SCATTER ADD KERNEL
	// ----------------------------------------------
		scatter_sum(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				SCATTER TRIAD KERNEL
	// ----------------------------------------------
		scatter_triad(stream_array_size, times, k, scalar);
	}
	
	// ----------------------------------------------
	// 				SCATTER VALIDATION
	// ----------------------------------------------
	scatter_validation(stream_array_size, scalar, is_validated, a, b, c);

// =================================================================================
//						CENTRAL VERSIONS OF THE KERNELS
// =================================================================================
	init_arrays(stream_array_size);

	for(int k = 0; k < NTIMES; k++) {
	// ----------------------------------------------
	// 				CENTRAL COPY KERNEL
	// ----------------------------------------------
		central_copy(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				CENTRAL SCALE KERNEL
	// ----------------------------------------------
		central_scale(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				CENTRAL ADD KERNEL
	// ----------------------------------------------
		central_sum(stream_array_size, times, k, scalar);
	// ----------------------------------------------
	// 				CENTRAL TRIAD KERNEL
	// ----------------------------------------------
		central_triad(stream_array_size, times, k, scalar);
	}
	
	// ----------------------------------------------
	// 				CENTRAL VALIDATION
	// ----------------------------------------------
	central_validation(stream_array_size, scalar, is_validated, a, b, c);
/*--------------------------------------------------------------------------------------
	// Calculate results
--------------------------------------------------------------------------------------*/
    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
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

/*--------------------------------------------------------------------------------------
	// Validate results
--------------------------------------------------------------------------------------*/
#ifdef INJECTERROR
	a[11] = 100.0 * a[11];
#endif
	checkSTREAMresults(is_validated);
    printf(HLINE);

	free(a);
	free(b);
	free(c);

	free(IDX1);
	free(IDX2);

    return 0;
}