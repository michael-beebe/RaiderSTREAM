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

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef SCALAR
#define SCALAR 0.42
#endif

#ifndef OFFSET
#   define OFFSET	0
#endif


# define HLINE "-------------------------------------------------------------\n"

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

# ifndef NUM_KERNELS
# define NUM_KERNELS 12
# endif

# ifndef NUM_ARRAYS
# define NUM_ARRAYS 3
# endif

STREAM_TYPE * restrict a, * restrict b, * restrict c;

size_t		array_elements, array_bytes, array_alignment;


static int a_idx[STREAM_ARRAY_SIZE];
static int b_idx[STREAM_ARRAY_SIZE];
static int c_idx[STREAM_ARRAY_SIZE];

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

extern void print_memory_usage(
	STREAM_TYPE a[STREAM_ARRAY_SIZE],
	STREAM_TYPE b[STREAM_ARRAY_SIZE],
	STREAM_TYPE c[STREAM_ARRAY_SIZE],
	STREAM_TYPE a_idx[STREAM_ARRAY_SIZE],
	STREAM_TYPE b_idx[STREAM_ARRAY_SIZE],
	STREAM_TYPE c_idx[STREAM_ARRAY_SIZE],
	double avgtime[NUM_KERNELS],
	double maxtime[NUM_KERNELS],
	double mintime[NUM_KERNELS],
	char label[NUM_KERNELS],
	double bytes[NUM_KERNELS],

    double *AvgErrByRank,
    

	int numranks,
	int BytesPerWord
);

int main() {
    int			quantum, checktick();
    int			BytesPerWord;
    int			i,k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t;
	double		*TimesByRank;
	double		t0,t1,tmin;
	int         rc, numranks, myrank;
	double *AvgErrByRank;

	array_elements = STREAM_ARRAY_SIZE / numranks;
    array_alignment = 64;

    array_bytes = array_elements * sizeof(STREAM_TYPE);

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


    printf("\n%d\n", BytesPerWord * ());
}

void print_memory_usage(
	STREAM_TYPE a[STREAM_ARRAY_SIZE],
	STREAM_TYPE b[STREAM_ARRAY_SIZE],
	STREAM_TYPE c[STREAM_ARRAY_SIZE],
	STREAM_TYPE a_idx[STREAM_ARRAY_SIZE],
	STREAM_TYPE b_idx[STREAM_ARRAY_SIZE],
	STREAM_TYPE c_idx[STREAM_ARRAY_SIZE],
	double avgtime[NUM_KERNELS],
	double maxtime[NUM_KERNELS],
	double mintime[NUM_KERNELS],
	char label[NUM_KERNELS],
	double bytes[NUM_KERNELS],

    double *AvgErrByRank,

	int numranks,
	int BytesPerWord
) {
	BytesPerWord = sizeof(STREAM_TYPE);
	// int totalmemory = \
	// 	BytesPerWord * ;


	printf(HLINE);
	printf("Totaly Memory Usage:\n");
	printf("\t\n");
	printf(HLINE);
}