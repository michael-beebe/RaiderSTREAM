#ifndef STREAM_OPENSHMEM_VALIDATION_H
#define STREAM_OPENSHMEM_VALIDATION_H

#include "stream_openshmem.h"

void check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors, ssize_t array_elements) {
  ssize_t i;
  int ierr = 0;

	if (abs(avg_err/exp_val) > epsilon) {
		(*errors)++;
		printf ("Failed Validation on array %s, AvgRelAbsErr > epsilon (%e)\n", label, epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", exp_val, avg_err, abs(avg_err/exp_val));
		ierr = 0;
		for (i=0; i<array_elements; i++) {
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

void central_check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors, ssize_t array_elements) {
  ssize_t i;

	if (abs(avg_err/exp_val) > epsilon) {
		(*errors)++;
		printf ("Failed Validation on array %s, AvgRelAbsErr > epsilon (%e)\n", label, epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", exp_val, avg_err, abs(avg_err/exp_val));
	}
}

int group_kernel_validation(ssize_t array_elements, STREAM_TYPE *AvgErrByRank, int numranks, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, KernelGroup group) {
    double epsilon;
    int err = 0;
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

	switch (group)
	{
	case CENTRAL:
		central_check_errors("a[]", a, aAvgErr, aj, epsilon, &err, array_elements);
		central_check_errors("b[]", b, bAvgErr, bj, epsilon, &err, array_elements);
		central_check_errors("c[]", c, cAvgErr, cj, epsilon, &err, array_elements);
		break;
	
	default:
		check_errors("a[]", a, aAvgErr, aj, epsilon, &err, array_elements);
		check_errors("b[]", b, bAvgErr, bj, epsilon, &err, array_elements);
		check_errors("c[]", c, cAvgErr, cj, epsilon, &err, array_elements);
		break;
	}

#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif

    return err;
}

void standard_errors(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t array_elements, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE *SumErr, STREAM_TYPE *AvgErr) {
	SumErr[0] = 0.0;
	SumErr[1] = 0.0;
	SumErr[2] = 0.0;

	for (ssize_t j = 0; j < array_elements; j++) {
		SumErr[0] += abs(a[j] - aj);
		SumErr[1] += abs(b[j] - bj);
		SumErr[2] += abs(c[j] - cj);
	}

	AvgErr[0] = SumErr[0] / (STREAM_TYPE) array_elements;
	AvgErr[1] = SumErr[1] / (STREAM_TYPE) array_elements;
	AvgErr[2] = SumErr[2] / (STREAM_TYPE) array_elements;
}

void central_errors(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t array_elements, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE *SumErr, STREAM_TYPE *AvgErr) {
	SumErr[0] = abs(a[0] - aj);
	SumErr[1] = abs(b[0] - bj);
	SumErr[2] = abs(c[0] - cj);

	AvgErr[0] = SumErr[0];
	AvgErr[1] = SumErr[1];
	AvgErr[2] = SumErr[2];
}


void validate_values(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t array_elements, STREAM_TYPE AvgErr[NUM_ARRAYS], STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, KernelGroup group) {
    STREAM_TYPE SumErr[NUM_ARRAYS];

	switch (group)
	{
	case CENTRAL:
		central_errors(aj, bj, cj, array_elements, a, b, c, SumErr, AvgErr);
		break;
	
	default:
		standard_errors(aj, bj, cj, array_elements, a, b, c, SumErr, AvgErr);
		break;
	}

#ifdef DEBUG
	printf("aSumErr= %f\t\t aAvgErr=%f\n", SumErr[0], AvgErr[0]);
	printf("bSumErr= %f\t\t bAvgErr=%f\n", SumErr[1], AvgErr[1]);
	printf("cSumErr= %f\t\t cAvgErr=%f\n", SumErr[2], AvgErr[2]);
#endif
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

    validate_values(aj, bj, cj, array_elements, AvgError, a, b, c, STREAM);

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
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj, STREAM);
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

    validate_values(aj, bj, cj, array_elements, AvgError, a, b, c, GATHER);
    
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
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj, GATHER);
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

    validate_values(aj, bj, cj, array_elements, AvgError, a, b, c, SCATTER);
    
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
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj, SCATTER);
        if(err == 0) {
            is_validated[SCATTER_COPY] = 1;
            is_validated[SCATTER_SCALE] = 1;
            is_validated[SCATTER_SUM] = 1;
            is_validated[SCATTER_TRIAD] = 1;
        }
    }  
}

void sg_validation(ssize_t array_elements, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, int myrank, int numranks, long *psync, STREAM_TYPE *AvgErrByRank, STREAM_TYPE *AvgError) {
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

    validate_values(aj, bj, cj, array_elements, AvgError, a, b, c, SG);
    
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
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj, SG);
        if(err == 0) {
            is_validated[SG_COPY] = 1;
            is_validated[SG_SCALE] = 1;
            is_validated[SG_SUM] = 1;
            is_validated[SG_TRIAD] = 1;
        }
    }  
}

void central_validation(ssize_t array_elements, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, int myrank, int numranks, long *psync, STREAM_TYPE *AvgErrByRank, STREAM_TYPE *AvgError) {
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

    validate_values(aj, bj, cj, array_elements, AvgError, a, b, c, CENTRAL);
    
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
        err = group_kernel_validation(array_elements, AvgErrByRank, numranks, a, b, c, aj, bj, cj, CENTRAL);
        if(err == 0) {
            is_validated[CENTRAL_COPY] = 1;
            is_validated[CENTRAL_SCALE] = 1;
            is_validated[CENTRAL_SUM] = 1;
            is_validated[CENTRAL_TRIAD] = 1;
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