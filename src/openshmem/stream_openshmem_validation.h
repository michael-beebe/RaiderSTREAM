#ifndef STREAM_OPENSHMEM_VALIDATION_H
#define STREAM_OPENSHMEM_VALIDATION_H

#include "stream_openshmem.h"

int group_kernel_validation(ssize_t array_elements, STREAM_TYPE *AvgErrByRank, int numranks, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj) {
    double epsilon;
    int err, ierr;
    STREAM_TYPE aAvgErr = 0.0, bAvgErr = 0.0, cAvgErr = 0.0;

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

    for (int k = 0; k < numranks; k++) {
		aAvgErr += AvgErrByRank[3*k + 0];
		bAvgErr += AvgErrByRank[3*k + 1];
		cAvgErr += AvgErrByRank[3*k + 2];
	}

    aAvgErr = aAvgErr / (STREAM_TYPE) numranks;
    bAvgErr = bAvgErr / (STREAM_TYPE) numranks;
    cAvgErr = cAvgErr / (STREAM_TYPE) numranks;

    err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (ssize_t j = 0; j < array_elements; j++) {
			if (abs(a[j]/aj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,aj,a[j],abs((aj-a[j])/aAvgErr));
				}
#endif
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
	}
	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (ssize_t j = 0; j < array_elements; j++) {
			if (abs(b[j]/bj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,bj,b[j],abs((bj-b[j])/bAvgErr));
				}
#endif
			}
		}
		printf("     For array b[], %d errors were found.\n",ierr);
	}
	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (ssize_t j = 0; j < array_elements; j++) {
			if (abs(c[j]/cj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,cj,c[j],abs((cj-c[j])/cAvgErr));
				}
#endif
			}
		}
		printf("     For array c[], %d errors were found.\n",ierr);
	}

#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif

    return err;
}

void validate_values(ssize_t array_elements, STREAM_TYPE AvgErr[NUM_ARRAYS], STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj) {
    STREAM_TYPE aSumErr, bSumErr, cSumErr;

    /* accumulate deltas between observed and expected results */
    aSumErr = 0.0;
    bSumErr = 0.0;
    cSumErr = 0.0;
	for (ssize_t j = 0; j < array_elements; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}

    AvgErr[0] = aSumErr / (STREAM_TYPE) array_elements;
	AvgErr[1] = bSumErr / (STREAM_TYPE) array_elements;
	AvgErr[2] = cSumErr / (STREAM_TYPE) array_elements;
}

void stream_validation(ssize_t array_elements, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, int myrank, int numranks, long *psync, STREAM_TYPE *AvgErrByRank, STREAM_TYPE *AvgError) {
    STREAM_TYPE aj,bj,cj;
    int BytesPerWord = sizeof(STREAM_TYPE);
    int err = 0;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

    /* now execute timing loop */
	scalar = SCALAR;
	for (int k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar*cj;
        cj = aj+bj;
        aj = bj+scalar*cj;
    }

    validate_values(array_elements, AvgError, a, b, c, aj, bj, cj);

    if(BytesPerWord == 4){
		shmem_fcollect32(AvgErrByRank, AvgError, NUM_ARRAYS, 0, 0, numranks, psync);
	}
	else if(BytesPerWord == 8){
		shmem_fcollect64(AvgErrByRank, AvgError, NUM_ARRAYS, 0, 0, numranks, psync);
	}
	else {
		printf("ERROR: sizeof(STREAM_TYPE) = %d\n", BytesPerWord);
		printf("ERROR: Please set STREAM_TYPE such that sizeof(STREAM_TYPE) = {4,8}\n");
		shmem_global_exit(1);
		exit(1);
	}

    if(myrank == 0) {
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj);
        if(err == 0) {
            is_validated[COPY] = 1;
            is_validated[SCALE] = 1;
            is_validated[SUM] = 1;
            is_validated[TRIAD] = 1;
        }
    }
}

void gather_validation(ssize_t array_elements, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, int myrank, int numranks, long *psync, STREAM_TYPE *AvgErrByRank, STREAM_TYPE *AvgError) {
    STREAM_TYPE aj,bj,cj;
    int BytesPerWord = sizeof(STREAM_TYPE);
    int err = 0;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

    /* now execute timing loop */
	scalar = SCALAR;
	for (int k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar*cj;
        cj = aj+bj;
        aj = bj+scalar*cj;
    }

    validate_values(array_elements, AvgError, a, b, c, aj, bj, cj);
    
    if(BytesPerWord == 4){
		shmem_fcollect32(AvgErrByRank, AvgError, NUM_ARRAYS, 0, 0, numranks, psync);
	}
	else if(BytesPerWord == 8){
		shmem_fcollect64(AvgErrByRank, AvgError, NUM_ARRAYS, 0, 0, numranks, psync);
	}
	else {
		printf("ERROR: sizeof(STREAM_TYPE) = %d\n", BytesPerWord);
		printf("ERROR: Please set STREAM_TYPE such that sizeof(STREAM_TYPE) = {4,8}\n");
		shmem_global_exit(1);
		exit(1);
	}

    if(myrank == 0) {
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj);
        if(err == 0) {
            is_validated[GATHER_COPY] = 1;
            is_validated[GATHER_SCALE] = 1;
            is_validated[GATHER_SUM] = 1;
            is_validated[GATHER_TRIAD] = 1;
        }
    }    
}

void scatter_validation(ssize_t array_elements, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, int myrank, int numranks, long *psync, STREAM_TYPE *AvgErrByRank, STREAM_TYPE *AvgError) {
    STREAM_TYPE aj,bj,cj;
    int BytesPerWord = sizeof(STREAM_TYPE);
    int err = 0;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

    /* now execute timing loop */
	scalar = SCALAR;
	for (int k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar*cj;
        cj = aj+bj;
        aj = bj+scalar*cj;
    }

    validate_values(array_elements, AvgError, a, b, c, aj, bj, cj);
    
    if(BytesPerWord == 4){
		shmem_fcollect32(AvgErrByRank, AvgError, NUM_ARRAYS, 0, 0, numranks, psync);
	}
	else if(BytesPerWord == 8){
		shmem_fcollect64(AvgErrByRank, AvgError, NUM_ARRAYS, 0, 0, numranks, psync);
	}
	else {
		printf("ERROR: sizeof(STREAM_TYPE) = %d\n", BytesPerWord);
		printf("ERROR: Please set STREAM_TYPE such that sizeof(STREAM_TYPE) = {4,8}\n");
		shmem_global_exit(1);
		exit(1);
	}

    if(myrank == 0) {
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj);
        if(err == 0) {
            is_validated[SCATTER_COPY] = 1;
            is_validated[SCATTER_SCALE] = 1;
            is_validated[SCATTER_SUM] = 1;
            is_validated[SCATTER_TRIAD] = 1;
        }
    }  
}

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

#endif