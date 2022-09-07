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
# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <time.h>
# include <getopt.h>

# define HLINE "---------------------------------------------------------------------------------------\n"

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef OFFSET
#   define OFFSET	0
#endif

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

extern void parse_opts(int argc, char **argv, ssize_t *stream_array_size);

extern double mysecond();

extern void init_random_idx_array(ssize_t *array, ssize_t nelems);
extern void init_read_idx_array(ssize_t *array, ssize_t nelems, char *filename);
extern void init_stream_array(STREAM_TYPE *array, ssize_t array_elements, STREAM_TYPE value);

extern void checkSTREAMresults(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t stream_array_size);
extern void check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors, ssize_t stream_array_size);

extern void print_info1(int BytesPerWord, ssize_t stream_array_size);
extern void print_timer_granularity(int quantum);
extern void print_info2(double t, int quantum);
extern void print_memory_usage(ssize_t stream_array_size);

#ifdef TUNED
void tuned_STREAM_Copy();
void tuned_STREAM_Scale(STREAM_TYPE scalar);
void tuned_STREAM_Add();
void tuned_STREAM_Triad(STREAM_TYPE scalar);
void tuned_STREAM_Copy_Gather();
void tuned_STREAM_Scale_Gather(STREAM_TYPE scalar);
void tuned_STREAM_Add_Gather();
void tuned_STREAM_Triad_Gather(STREAM_TYPE scalar);
void tuned_STREAM_Copy_Scatter();
void tuned_STREAM_Scale_Scatter(STREAM_TYPE scalar);
void tuned_STREAM_Add_Scatter();
void tuned_STREAM_Triad_Scatter(STREAM_TYPE scalar);
#endif

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif


int main(int argc, char *argv[])
{
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
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(int))) * stream_array_size), // GATHER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(int))) * stream_array_size), // GATHER Scale
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(int))) * stream_array_size), // GATHER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(int))) * stream_array_size), // GATHER Triad
		// Scatter Kernels
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(int))) * stream_array_size), // SCATTER copy
		(((2 * sizeof(STREAM_TYPE)) + (1 * sizeof(int))) * stream_array_size), // SCATTER Scale
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(int))) * stream_array_size), // SCATTER Add
		(((3 * sizeof(STREAM_TYPE)) + (2 * sizeof(int))) * stream_array_size), // SCATTER Triad
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

// =================================================================================
    		/*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
// =================================================================================
    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
// ----------------------------------------------
// 				  COPY KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    c[j] = a[j];
#endif
	t1 = mysecond();
	times[0][k] = t1 - t0;

// ----------------------------------------------
// 		 	     SCALE KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    b[j] = scalar * c[j];
#endif
	t1 = mysecond();
	times[1][k] = t1 - t0;
// ----------------------------------------------
// 				 ADD KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    c[j] = a[j] + b[j];
#endif
	t1 = mysecond();
	times[2][k] = t1 - t0;
// ----------------------------------------------
//				TRIAD KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    a[j] = b[j] + scalar * c[j];
#endif
	t1 = mysecond();
	times[3][k] = t1 - t0;

// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
// ----------------------------------------------
// 				GATHER COPY KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Copy_Gather(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[j] = a[IDX1[j]];
#endif
	t1 = mysecond();
	times[4][k] = t1 - t0;

// ----------------------------------------------
// 				GATHER SCALE KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Scale_Gather(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[j] = scalar * c[IDX2[j]];
#endif
	t1 = mysecond();
	times[5][k] = t1 - t0;

// ----------------------------------------------
// 				GATHER ADD KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Add_Gather();
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[j] = a[IDX1[j]] + b[IDX2[j]];
#endif
	t1 = mysecond();
	times[6][k] = t1 - t0;

// ----------------------------------------------
// 			   GATHER TRIAD KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Triad_Gather(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[j] = b[IDX1[j]] + scalar * c[IDX2[j]];
#endif
	t1 = mysecond();
	times[7][k] = t1 - t0;

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
// ----------------------------------------------
// 				SCATTER COPY KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Copy_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[IDX1[j]] = a[j];
#endif
	t1 = mysecond();
	times[8][k] = t1 - t0;

// ----------------------------------------------
// 				SCATTER SCALE KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Scale_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[IDX2[j]] = scalar * c[j];
#endif
	t1 = mysecond();
	times[9][k] = t1 - t0;

// ----------------------------------------------
// 				SCATTER ADD KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_ADD_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[IDX1[j]] = a[j] + b[j];
#endif
	t1 = mysecond();
	times[10][k] = t1 - t0;

// ----------------------------------------------
// 				SCATTER TRIAD KERNEL
// ----------------------------------------------
	t0 = mysecond();
#ifdef TUNED
		tuned_STREAM_Triad_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
    a[IDX2[j]] = b[j] + scalar * c[j];
#endif
	t1 = mysecond();
	times[11][k] = t1 - t0;
}

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
	checkSTREAMresults(a, b, c, stream_array_size);
    printf(HLINE);

	free(a);
	free(b);
	free(c);

	free(IDX1);
	free(IDX2);

    return 0;
}






# define	M	20
int checktick()
{
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
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

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */
double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

/*--------------------------------------------------------------------------------------
 - Initializes provided array with random indices within data array
    bounds. Forces a one-to-one mapping from available data array indices
    to utilized indices in index array. This simplifies the scatter kernel
    verification process and precludes the need for atomic operations.
--------------------------------------------------------------------------------------*/
void init_random_idx_array(ssize_t *array, ssize_t nelems) {
	int success;
	ssize_t i, idx;

	// Array to track used indices
	char* flags = (char*) malloc(sizeof(char)*nelems);
	for(i = 0; i < nelems; i++){
		flags[i] = 0;
	}

	// Iterate and fill each element of the idx array
	for (i = 0; i < nelems; i++) {
		success = 0;
		while(success == 0){
			idx = ((ssize_t) rand()) % nelems;
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

/*--------------------------------------------------------------------------------------
 - Populate specified array with the specified value
--------------------------------------------------------------------------------------*/
void init_stream_array(STREAM_TYPE *array, ssize_t array_elements, STREAM_TYPE value) {
    #pragma omp parallel for
    for (ssize_t i = 0; i < array_elements; i++) {
        array[i] = value;
    }
}


/*--------------------------------------------------------------------------------------
 - Check STREAM results to ensure acuracy
--------------------------------------------------------------------------------------*/
#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

void checkSTREAMresults(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, ssize_t stream_array_size)
{
	STREAM_TYPE aj,bj,cj;
	STREAM_TYPE aSumErr, bSumErr, cSumErr;
	STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;

	STREAM_TYPE scalar;

	double epsilon;
	ssize_t	j;
	int	k,err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;

    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

  /* now execute timing loop  */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++){
		// Sequential kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
		// Gather kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
		// Scatter kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
  }

  

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0, bSumErr = 0.0, cSumErr = 0.0;
	for (j=0; j<stream_array_size; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}

	aAvgErr = aSumErr / (STREAM_TYPE) stream_array_size;
	bAvgErr = bSumErr / (STREAM_TYPE) stream_array_size;
	cAvgErr = cSumErr / (STREAM_TYPE) stream_array_size;

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

#ifdef DEBUG
	printf("aSumErr= %f\t\t aAvgErr=%f\n", aSumErr, aAvgErr);
	printf("bSumErr= %f\t\t bAvgErr=%f\n", bSumErr, bAvgErr);
	printf("cSumErr= %f\t\t cAvgErr=%f\n", cSumErr, cAvgErr);
#endif


 // Check errors on each array
  check_errors("a[]", a, aAvgErr, aj, epsilon, &err, stream_array_size);
  check_errors("b[]", b, bAvgErr, bj, epsilon, &err, stream_array_size);
  check_errors("c[]", c, cAvgErr, cj, epsilon, &err, stream_array_size);

	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all arrays\n", epsilon);
	}
#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

/* Checks error results against epsilon and prints debug info */
void check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors, ssize_t stream_array_size) {
  ssize_t i;
  int ierr = 0;

	if (abs(avg_err/exp_val) > epsilon) {
		(*errors)++;
		printf ("Failed Validation on array %s, AvgRelAbsErr > epsilon (%e)\n", label, epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", exp_val, avg_err, abs(avg_err/exp_val));
		ierr = 0;
		for (i=0; i<stream_array_size; i++) {
			if (abs(array[i]/exp_val-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array %s: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						label, i, exp_val, array[i], abs((exp_val-array[i])/avg_err));
				}
#endif
			}
		}
		printf("     For array %s, %d errors were found.\n", label, ierr);
	}
}

/*--------------------------------------------------------------------------------------
 - Functions for printing initial system information and so forth
--------------------------------------------------------------------------------------*/
void print_info1(int BytesPerWord, ssize_t stream_array_size) {
    printf(HLINE);
    printf("RaiderSTREAM\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef N
    printf("*****  WARNING: ******\n");
    printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
    printf("      This version of the code uses the preprocesor variable stream_array_size to control the array size\n");
    printf("      Reverting to default value of stream_array_size=%llu\n",(unsigned long long) stream_array_size);
    printf("*****  WARNING: ******\n");
#endif

    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) stream_array_size, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n",
	BytesPerWord * ( (double) stream_array_size / 1024.0/1024.0),
	BytesPerWord * ( (double) stream_array_size / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) stream_array_size / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) stream_array_size / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n");
    printf(" will be used to compute the reported bandwidth.\n");
}

void print_timer_granularity(int quantum) {
    printf(HLINE);
    if  ( (quantum = checktick()) >= 1)
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }
}

void print_info2(double t, int quantum) {
    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
	// printf("   (= %d timer ticks)\n", (int) (t/quantum) );
    printf("   (= %d clock ticks)\n", (int) (t) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
}

void print_memory_usage(ssize_t stream_array_size) {
	unsigned long totalMemory = \
		(sizeof(STREAM_TYPE) * (stream_array_size)) + 	// a[]
		(sizeof(STREAM_TYPE) * (stream_array_size)) + 	// b[]
		(sizeof(STREAM_TYPE) * (stream_array_size)) + 	// c[]
		(sizeof(ssize_t) * (stream_array_size)) + 			// a_idx[]
		(sizeof(ssize_t) * (stream_array_size)) + 			// b_idx[]
		(sizeof(ssize_t) * (stream_array_size)) + 			// c_idx[]
		(sizeof(double) * NUM_KERNELS) + 				// avgtime[]
		(sizeof(double) * NUM_KERNELS) + 				// maxtime[]
		(sizeof(double) * NUM_KERNELS) + 				// mintime[]
		(sizeof(char) * NUM_KERNELS) +					// label[]
		(sizeof(double) * NUM_KERNELS);					// bytes[]

#ifdef VERBOSE // if -DVERBOSE enabled break down memory usage by each array
	printf("---------------------------------\n");
	printf("  VERBOSE Memory Breakdown\n");
	printf("---------------------------------\n");
	printf("a[]:\t\t%.2f MB\n", (sizeof(STREAM_TYPE) * (stream_array_size)) / 1024.0 / 1024.0);
	printf("b[]:\t\t%.2f MB\n", (sizeof(STREAM_TYPE) * (stream_array_size)) / 1024.0 / 1024.0);
	printf("c[]:\t\t%.2f MB\n", (sizeof(STREAM_TYPE) * (stream_array_size)) / 1024.0 / 1024.0);
	printf("a_idx[]:\t%.2f MB\n", (sizeof(int) * (stream_array_size)) / 1024.0 / 1024.0);
	printf("b_idx[]:\t%.2f MB\n", (sizeof(int) * (stream_array_size)) / 1024.0 / 1024.0);
	printf("c_idx[]:\t%.2f MB\n", (sizeof(int) * (stream_array_size)) / 1024.0 / 1024.0);
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

void parse_opts(int argc, char **argv, ssize_t *stream_array_size) {
    int option;
    while( (option = getopt(argc, argv, "n:t:h")) != -1 ) {
        switch (option) {
            case 'n':
                *stream_array_size = atoi(optarg);
                break;
            case 'h':
                printf("Usage: -n <stream_array_size>\n");
                exit(2);
			
        }
    }
}

/* stubs for "tuned" versions of the kernels */
#ifdef TUNED

ssize_t stream_array_size;
parse_opts(argc, argv, &stream_array_size);

// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
void tuned_STREAM_Copy()
{
	ssize_t j;
#pragma omp parallel for
    for (j=0; j<stream_array_size; j++)
        c[j] = a[j];
}

void tuned_STREAM_Scale(STREAM_TYPE scalar)
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    b[j] = scalar*c[j];
}

void tuned_STREAM_Add()
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    a[j] = b[j]+scalar*c[j];
}
// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Gather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[j] = a[a_idx[j]];
}

void tuned_STREAM_Scale_Gather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[j] = scalar * c[c_idx[j]];
}

void tuned_STREAM_Add_Gather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[j] = a[a_idx[j]] + b[b_idx[j]];
}

void tuned_STREAM_Triad_Gather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[j] = b[b_idx[j]] + scalar * c[c_idx[j]];
}

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Scatter() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[c_idx[j]] = a[j];
}

void tuned_STREAM_Scale_Scatter(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[b_idx[j]] = scalar * c[j];
}

void tuned_STREAM_Add_Scatter() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[a_idx[j]] = a[j] + b[j];
}

void tuned_STREAM_Triad_Scatter(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[a_idx[j]] = b[j] + scalar * c[j];
}
/* end of stubs for the "tuned" versions of the kernels */
#endif

