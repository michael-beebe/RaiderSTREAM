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

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

# ifndef NUM_KERNELS
# define NUM_KERNELS 12
# endif

# ifndef NUM_ARRAYS
# define NUM_ARRAYS 3
# endif

size_t		array_elements, array_bytes, array_alignment;

STREAM_TYPE * restrict a, * restrict b, * restrict c;

int * restrict a_idx, * restrict b_idx, * restrict c_idx;

// static int a_idx[array_elements];
// static int b_idx[array_elements];
// static int c_idx[array_elements];


extern void init_idx_array(int *array, int nelems);

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

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
	STREAM_TYPE *AvgErrByRank;

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
    - Initialize the idx arrays on all PEs and populate them with random values ranging
        from 0 - array_elements
--------------------------------------------------------------------------------------*/
    srand(time(0));
    init_idx_array(a_idx, array_elements);
    init_idx_array(b_idx, array_elements);
    init_idx_array(c_idx, array_elements);

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
    // Populate STREAM arrays on all ranks
--------------------------------------------------------------------------------------*/
#pragma omp parallel for
    for (j=0; j<array_elements; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}
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
