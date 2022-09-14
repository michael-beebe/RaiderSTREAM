#ifndef STREAM_OPENMP_H
#define STREAM_OPENMP_H

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

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

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
	SCATTER_TRIAD
};

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
	"SCATTER_TRIAD"
};

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

#endif