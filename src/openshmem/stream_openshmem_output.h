#ifndef STREAM_OPENSHMEM_OUTPUT_H
#define STREAM_OPENSHMEM_OUTPUT_H

#include "stream_openshmem.h"

/*--------------------------------------------------------------------------------------
- Initialize array to store labels for the benchmark kernels.
- This is for
--------------------------------------------------------------------------------------*/
static char	*label[NUM_KERNELS] = {
    "STREAM Copy:\t", "STREAM Scale:\t",
    "STREAM Add:\t", "STREAM Triad:\t",
	"GATHER Copy:\t", "GATHER Scale:\t",
	"GATHER Add:\t", "GATHER Triad:\t",
	"SCATTER Copy:\t", "SCATTER Scale:\t",
	"SCATTER Add:\t", "SCATTER Triad:\t",
	"SG Copy:\t", "SG Scale:\t",
	"SG Add:\t\t", "SG Triad:\t",
	"CENTRAL Copy:\t", "CENTRAL Scale:\t",
	"CENTRAL Add:\t", "CENTRAL Triad:\t"
};


void print_info1(int BytesPerWord, int numranks, ssize_t array_elements, ssize_t stream_array_size) {
		printf(HLINE);
		printf("RaiderSTREAM\n");
		printf(HLINE);
		//BytesPerWord = sizeof(STREAM_TYPE);
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

		if (OFFSET != 0) {
			printf("*****  WARNING: ******\n");
			printf("   This version ignores the OFFSET parameter.\n");
			printf("*****  WARNING: ******\n");
		}

		printf("Total Aggregate Array size = %llu (elements)\n" , (unsigned long long) stream_array_size);
		printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n",
			BytesPerWord * ( (double) stream_array_size / 1024.0/1024.0),
			BytesPerWord * ( (double) stream_array_size / 1024.0/1024.0/1024.0));
		printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
			(3.0 * BytesPerWord) * ( (double) stream_array_size / 1024.0/1024.),
			(3.0 * BytesPerWord) * ( (double) stream_array_size / 1024.0/1024./1024.));
		printf("Data is distributed across %d SHMEM ranks\n",numranks);
		printf("   Array size per SHMEM rank = %llu (elements)\n" , (unsigned long long) array_elements);
		printf("   Memory per array per SHMEM rank = %.1f MiB (= %.1f GiB).\n",
			BytesPerWord * ( (double) array_elements / 1024.0/1024.0),
			BytesPerWord * ( (double) array_elements / 1024.0/1024.0/1024.0));
		printf("   Total memory per SHMEM rank = %.1f MiB (= %.1f GiB).\n",
			(3.0 * BytesPerWord) * ( (double) array_elements / 1024.0/1024.),
			(3.0 * BytesPerWord) * ( (double) array_elements / 1024.0/1024./1024.));

		printf(HLINE);
		printf("Each kernel will be executed %d times.\n", NTIMES);
		printf(" The *best* time for each kernel (excluding the first iteration)\n");
		printf(" will be used to compute the reported bandwidth.\n");
		printf("The SCALAR value used for this run is %f\n",SCALAR);
}

void print_timer_granularity(int quantum) {
    if  ( (quantum = checktick()) >= 1)
        printf("Your timer granularity/precision appears to be "
            "%d microseconds.\n", quantum);
    else {
        printf("Your timer granularity appears to be "
            "less than one microsecond.\n");
        quantum = 1;
    }
}

void print_info2(double t, double t0, double t1, int quantum) {
    printf("Each test below will take on the order"
    " of %d microseconds.\n", (int) t  );
    // printf("   (= %d timer ticks)\n", (int) (t/quantum) );
    printf("   (= %d timer ticks)\n", (int) (t) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 timer ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    #ifdef VERBOSE
        t1 = mysecond();
        printf("VERBOSE: total setup time for rank 0 = %f seconds\n", t1 - t0);
        printf(HLINE);
    #endif
}

void print_memory_usage(int numranks, ssize_t STREAM_ARRAY_SIZE) {
	unsigned long totalMemory = \
		((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) + 	// a[]
		((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) + 	// b[]
		((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) + 	// c[]
		((sizeof(ssize_t) * (STREAM_ARRAY_SIZE)) * numranks) + 			// IDX1[]
		((sizeof(ssize_t) * (STREAM_ARRAY_SIZE)) * numranks) + 			// IDX2[]
		((sizeof(double) * NUM_KERNELS) * numranks) + 				// avgtime[]
		((sizeof(double) * NUM_KERNELS) * numranks) + 				// maxtime[]
		((sizeof(double) * NUM_KERNELS) * numranks) + 				// mintime[]
		((sizeof(char) * NUM_KERNELS) * numranks) +					// label[]
		((sizeof(double) * NUM_KERNELS) * numranks);				// bytes[]

#ifdef VERBOSE // if -DVERBOSE enabled break down memory usage by each array
	printf("-----------------------------------------\n");
	printf("     VERBOSE Memory Breakdown\n");
	printf("-----------------------------------------\n");
	printf("a[]:\t\t%.2f MB\n", ((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("b[]:\t\t%.2f MB\n", ((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("c[]:\t\t%.2f MB\n", ((sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("IDX1[]:\t%.2f MB\n", ((sizeof(ssize_t) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("IDX2[]:\t%.2f MB\n", ((sizeof(ssize_t) * (STREAM_ARRAY_SIZE)) * numranks) / 1024.0 / 1024.0);
	printf("avgtime[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("maxtime[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("mintime[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("label[]:\t%lu B\n", ((sizeof(char) * NUM_KERNELS) * numranks));
	printf("bytes[]:\t%lu B\n", ((sizeof(double) * NUM_KERNELS) * numranks));
	printf("-----------------------------------------\n");
	printf("Total Memory Allocated Across All Ranks: %.2f MB\n", totalMemory / 1024.0 / 1024.0);
	printf("-----------------------------------------\n");
#else
	printf("Totaly Memory Allocated Across All Ranks: %.2f MB\n", totalMemory / 1024.0 / 1024.0);
#endif
	printf(HLINE);
}

#endif // STREAM_OPENSHMEM_OUTPUT_H