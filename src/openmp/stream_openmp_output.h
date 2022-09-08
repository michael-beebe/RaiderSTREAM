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

#endif