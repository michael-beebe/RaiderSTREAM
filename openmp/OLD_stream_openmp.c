/*-----------------------------------------------------------------------*/
/* Program: STREAM                                                       */
/* Revision: $Id: stream.c,v 5.10 2013/01/17 16:01:06 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2013: John D. McCalpin                                 */
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

// #define DEBUG 1

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
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
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

/*  Users are allowed to modify the "OFFSET" variable, which *may* change the
 *         relative alignment of the arrays (though compilers may change the
 *         effective offset by making the arrays non-contiguous on some systems).
 *      Use of non-zero values for OFFSET can be especially helpful if the
 *         STREAM_ARRAY_SIZE is set to a value close to a large power of 2.
 *      OFFSET can also be set on the compile line without changing the source
 *         code using, for example, "-DOFFSET=56".
 */
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

# define HLINE "---------------------------------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

//===============================================================
# ifndef NUM_KERNELS
# define NUM_KERNELS 12
# endif
//===============================================================
// Create a, b, c arrays for original kernels
static STREAM_TYPE a[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE b[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE c[STREAM_ARRAY_SIZE+OFFSET];

// Create gather_a, gather_b, gather_c arrays for GATHER kernels
static STREAM_TYPE gather_a[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE gather_b[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE gather_c[STREAM_ARRAY_SIZE+OFFSET];

// Create scatter_a, scatter_b, scatter_c for SCATTER kernels
static STREAM_TYPE scatter_a[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE scatter_b[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE scatter_c[STREAM_ARRAY_SIZE+OFFSET];

//===============================================================
// Create idx arrays (which will be used by gather/scatter kernels)
static int a_idx[STREAM_ARRAY_SIZE+OFFSET];
static int b_idx[STREAM_ARRAY_SIZE+OFFSET];
static int c_idx[STREAM_ARRAY_SIZE+OFFSET];

//===============================================================
// Initiate metrics for performance benchmarking
static double avgtime[NUM_KERNELS] = {0};
static double maxtime[NUM_KERNELS] = {0};
static double mintime[NUM_KERNELS];
static char	*label[NUM_KERNELS] = {
  "Copy:\t\t", "Scale:\t\t",
  "Add:\t\t", "Triad:\t\t",
	"GATHER Copy:\t", "GATHER Scale:\t",
	"GATHER Add:\t", "GATHER Triad:\t",
	"SCATTER Copy:\t", "SCATTER Scale:\t",
	"SCATTER Add:\t", "SCATTER Triad:\t"
};

static double	bytes[NUM_KERNELS] = {
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

double mysecond();
void checkSTREAMresults();
void init_idx_array(int *array, int nelems);
void check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors);
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

int main()
{
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[NUM_KERNELS][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */
    // printf(HLINE);
    // printf("STREAM version $Revision: 5.10 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef N
    printf("*****  WARNING: ******\n");
    printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
    printf("      This version of the code uses the preprocesor variable STREAM_ARRAY_SIZE to control the array size\n");
    printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",(unsigned long long) STREAM_ARRAY_SIZE);
    printf("*****  WARNING: ******\n");
#endif

    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n",
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n");
    printf(" will be used to compute the reported bandwidth.\n");

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

	//===============================================================
    /* --- SETUP --- populate the idx arrays --- */
    srand(time(0));
    init_idx_array(a_idx, STREAM_ARRAY_SIZE);
    init_idx_array(b_idx, STREAM_ARRAY_SIZE);
    init_idx_array(c_idx, STREAM_ARRAY_SIZE);
	//===============================================================

    /* Get initial value for system clock. */
	#pragma omp parallel for private (j)
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
      a[j] = 1.0;
      b[j] = 2.0;
      c[j] = 0.0;

      gather_a[j] = 1.0;
      gather_b[j] = 2.0;
      gather_c[j] = 0.0;

      scatter_a[j] = 1.0;
      scatter_b[j] = 2.0;
      scatter_c[j] = 0.0;
    }

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1)
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }
    t = mysecond();

#pragma omp parallel for private (j)
    for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
  		a[j] = 2.0E0 * a[j];
  		gather_a[j] = 2.0E0 * gather_a[j];
  		scatter_a[j] = 2.0E0 * scatter_a[j];
	}

    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);

    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
// ----------------------------------------------
// 				  COPY KERNEL
// ----------------------------------------------
	times[0][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j];
#endif
	times[0][k] = mysecond() - times[0][k];

// ----------------------------------------------
// 		 	     SCALE KERNEL
// ----------------------------------------------
	times[1][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    b[j] = scalar * c[j];
#endif
	times[1][k] = mysecond() - times[1][k];
// ----------------------------------------------
// 				 ADD KERNEL
// ----------------------------------------------
	times[2][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j] + b[j];
#endif
	times[2][k] = mysecond() - times[2][k];
// ----------------------------------------------
//				TRIAD KERNEL
// ----------------------------------------------
	times[3][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    a[j] = b[j] + scalar * c[j];
#endif
	times[3][k] = mysecond() - times[3][k];

// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
// ----------------------------------------------
// 				GATHER COPY KERNEL
// ----------------------------------------------
	times[4][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Copy_Gather(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		gather_c[j] = gather_a[a_idx[j]];
#endif
	times[4][k] = mysecond() - times[4][k];
// ----------------------------------------------
// 				GATHER SCALE KERNEL
// ----------------------------------------------
	times[5][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Scale_Gather(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		gather_b[j] = scalar * gather_c[c_idx[j]];
#endif
	times[5][k] = mysecond() - times[5][k];

// ----------------------------------------------
// 				GATHER ADD KERNEL
// ----------------------------------------------
	times[6][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Add_Gather();
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		gather_c[j] = gather_a[a_idx[j]] + gather_b[b_idx[j]];
#endif
	times[6][k] = mysecond() - times[6][k];
// ----------------------------------------------
// 			   GATHER TRIAD KERNEL
// ----------------------------------------------
	times[7][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Triad_Gather(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		gather_a[j] = gather_b[b_idx[j]] + scalar * gather_c[c_idx[j]];
#endif
	times[7][k] = mysecond() - times[7][k];

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
// ----------------------------------------------
// 				SCATTER COPY KERNEL
// ----------------------------------------------
	times[8][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Copy_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		scatter_c[c_idx[j]] = scatter_a[j];
#endif
	times[8][k] = mysecond() - times[8][k];

// ----------------------------------------------
// 				SCATTER SCALE KERNEL
// ----------------------------------------------
	times[9][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Scale_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		scatter_b[b_idx[j]] = scalar * scatter_c[j];
#endif
	times[9][k] = mysecond() - times[9][k];

// ----------------------------------------------
// 				SCATTER ADD KERNEL
// ----------------------------------------------
	times[10][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_ADD_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
		scatter_c[c_idx[j]] = scatter_a[j] + scatter_b[j];
#endif
	times[10][k] = mysecond() - times[10][k];

// ----------------------------------------------
// 				SCATTER TRIAD KERNEL
// ----------------------------------------------
	times[11][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Triad_Scatter(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
    scatter_a[a_idx[j]] = scatter_b[j] + scalar * scatter_c[j];
#endif
	times[11][k] = mysecond() - times[11][k];
}

    /*	--- SUMMARY --- */

	for (int i = 0; i < NUM_KERNELS; i++) {
		mintime[i] = FLT_MAX;
	}

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<NUM_KERNELS; j++)
	    {
			avgtime[j] = avgtime[j] + times[j][k];
			mintime[j] = MIN(mintime[j], times[j][k]);
			maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

    printf("Function\tBest Rate MB/s\t   Avg time\t   Min time\t   Max time\n");
    for (j=0; j<NUM_KERNELS; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		printf("%s%12.1f\t%11.6f\t%11.6f\t%11.6f\n",
			label[j],
			1.0E-06 * bytes[j]/mintime[j],
			avgtime[j],
			mintime[j],
			maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    // checkSTREAMresults();
    printf(HLINE);

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

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

void checkSTREAMresults()
{
	STREAM_TYPE aj,bj,cj;

	STREAM_TYPE aSumErr, bSumErr, cSumErr;
	STREAM_TYPE gather_aSumErr, gather_bSumErr, gather_cSumErr;
	STREAM_TYPE scatter_aSumErr, scatter_bSumErr, scatter_cSumErr;

	STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;
	STREAM_TYPE gather_aAvgErr, gather_bAvgErr, gather_cAvgErr;
	STREAM_TYPE scatter_aAvgErr, scatter_bAvgErr, scatter_cAvgErr;

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
		// Original
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
  }

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0, bSumErr = 0.0, cSumErr = 0.0;
	gather_aSumErr = 0.0, gather_bSumErr = 0.0, gather_cSumErr = 0.0;
	scatter_aSumErr = 0.0, scatter_bSumErr = 0.0, scatter_cSumErr = 0.0;
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
    	// Original kernels
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);

		// Gather kernels
		gather_aSumErr += abs(gather_a[j] - aj);
		gather_bSumErr += abs(gather_b[j] - bj);
		gather_cSumErr += abs(gather_c[j] - cj);

		// Scatter kernels
		scatter_aSumErr += abs(scatter_a[j] - aj);
		scatter_bSumErr += abs(scatter_b[j] - bj);
		scatter_cSumErr += abs(scatter_c[j] - cj);
	}

	aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

	gather_aAvgErr = gather_aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	gather_bAvgErr = gather_bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	gather_cAvgErr = gather_cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

	scatter_aAvgErr = scatter_aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	scatter_bAvgErr = scatter_bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	scatter_cAvgErr = scatter_cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

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

  printf("gather_aSumErr= %f\t gather_aAvgErr=%f\n", gather_aSumErr, gather_aAvgErr);
  printf("gather_bSumErr= %f\t gather_bAvgErr=%f\n", gather_bSumErr, gather_bAvgErr);
  printf("gather_cSumErr= %f\t gather_cAvgErr=%f\n", gather_cSumErr, gather_cAvgErr);

  printf("scatter_aSumErr= %f\t scatter_aAvgErr=%f\n", scatter_aSumErr, scatter_aAvgErr);
  printf("scatter_bSumErr= %f\t scatter_bAvgErr=%f\n", scatter_bSumErr, scatter_bAvgErr);
  printf("scatter_cSumErr= %f\t scatter_cAvgErr=%f\n", scatter_cSumErr, scatter_cAvgErr);
#endif

  // Original kernels
  check_errors("a[]", a, aAvgErr, aj, epsilon, &err);
  check_errors("b[]", b, bAvgErr, bj, epsilon, &err);
  check_errors("c[]", c, cAvgErr, cj, epsilon, &err);

  // Gather kernels
  check_errors("gather_a[]", gather_a, gather_aAvgErr, aj, epsilon, &err);
  check_errors("gather_b[]", gather_b, gather_bAvgErr, bj, epsilon, &err);
  check_errors("gather_c[]", gather_c, gather_cAvgErr, cj, epsilon, &err);

  // Scatter kernels
  check_errors("scatter_a[]", scatter_a, scatter_aAvgErr, aj, epsilon, &err);
  check_errors("scatter_b[]", scatter_b, scatter_bAvgErr, bj, epsilon, &err);
  check_errors("scatter_c[]", scatter_c, scatter_cAvgErr, cj, epsilon, &err);

	if (err == 0) {
		printf ("\nSolution Validates: avg error less than %e on all nine arrays\n",epsilon);
	}
#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Observed gather_a(1), gather_b(1), gather_c(1): %f %f %f \n",gather_a[1],gather_b[1],gather_c[1]);
	printf ("    Observed scatter_a(1), scatter_b(1), scatter_c(1): %f %f %f \n",scatter_a[1],scatter_b[1],scatter_c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
	printf ("    Rel Errors on gather_a, gather_b, gather_c:     %e %e %e \n",abs(gather_aAvgErr/aj),abs(gather_bAvgErr/bj),abs(gather_cAvgErr/cj));
	printf ("    Rel Errors on scatter_a, scatter_b, scatter_c:     %e %e %e \n",abs(scatter_aAvgErr/aj),abs(scatter_bAvgErr/bj),abs(scatter_cAvgErr/cj));
#endif
}

/*
 * Initializes provided array with random indices within data array
 * bounds. Forces a one-to-one mapping from available data array indices
 * to utilized indices in index array. This simplifies the scatter kernel
 * verification process and precludes the need for atomic operations.
 */
void init_idx_array(int *array, int nelems){
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

/* Checks error results against epsilon and prints debug info */
void check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors) {
  int i;
  int ierr = 0;

	if (abs(avg_err/exp_val) > epsilon) {
		(*errors)++;
		printf ("Failed Validation on array %s, AvgRelAbsErr > epsilon (%e)\n", label, epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", exp_val, avg_err, abs(avg_err/exp_val));
		ierr = 0;
		for (i=0; i<STREAM_ARRAY_SIZE; i++) {
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

