# ifndef STREAM_OPENMP_OUTPUT_H
# define STREAM_OPENMP_OUTPUT_H

#include "stream_openmp.h"

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

extern void checkSTREAMresults(int *is_validated);
extern void check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors, ssize_t stream_array_size);

extern void print_info1(int BytesPerWord, ssize_t stream_array_size);
extern void print_timer_granularity(int quantum);
extern void print_info2(double t, int quantum);
extern void print_memory_usage(ssize_t stream_array_size);

#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

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

void checkSTREAMresults(int *is_validated)
{
	int incorrect = 0;
	double epsilon;

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

	for(int i = 0; i < NUM_KERNELS; i++) {
		if(is_validated[i] != 1) {
			printf("Kernel %s validation not correct\n", kernel_map[i]);
			incorrect++;
		}
	}

	if(incorrect == 0) {
		printf ("Solution Validates: avg error less than %e on all arrays\n", 	epsilon);
	}
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
		(sizeof(ssize_t) * (stream_array_size)) + 		// IDX1
		(sizeof(ssize_t) * (stream_array_size)) + 		// IDX2
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
	printf("IDX1:\t%.2f MB\n", (sizeof(int) * (stream_array_size)) / 1024.0 / 1024.0);
	printf("IDX2:\t%.2f MB\n", (sizeof(int) * (stream_array_size)) / 1024.0 / 1024.0);
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

#endif