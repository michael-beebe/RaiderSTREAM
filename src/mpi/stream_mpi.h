#ifndef STREAM_MPI_H
#define STREAM_MPI_H

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

# define HLINE "------------------------------------------------------------------------------------------------\n"

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
# define NUM_KERNELS 16
# endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

/*--------------------------------------------------------------------------------------
- Specifies the total number of stream arrays used in the main loop
--------------------------------------------------------------------------------------*/
# ifndef NUM_ARRAYS
# define NUM_ARRAYS 3
# endif

enum Kernels {
	COPY,
	SCALE,
	SUM,
	TRIAD,
	GATHER_COPY,
	GATHER_SCALE,
	GATHER_SUM,
	GATHER_TRIAD,
	SCATTER_COPY,
	SCATTER_SCALE,
	SCATTER_SUM,
	SCATTER_TRIAD,
	CENTRAL_COPY,
	CENTRAL_SCALE,
	CENTRAL_SUM,
	CENTRAL_TRIAD
};

typedef enum {
	STREAM,
	GATHER,
	SCATTER,
	CENTRAL
} KernelGroup;

static char *kernel_map[NUM_KERNELS] = {
	"COPY",
	"SCALE",
	"SUM",
	"TRIAD",
	"GATHER_COPY",
	"GATHER_SCALE",
	"GATHER_SUM",
	"GATHER_TRIAD",
	"SCATTER_COPY",
	"SCATTER_SCALE",
	"SCATTER_SUM",
	"SCATTER_TRIAD",
	"CENTRAL_COPY",
	"CENTRAL_SCALE",
	"CENTRAL_SUM",
	"CENTRAL_TRIAD"
};


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

#endif