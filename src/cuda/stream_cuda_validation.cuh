#ifndef STREAM_CUDA_VALIDATION_CUH
#define STREAM_CUDA_VALIDATION_CUH

#include "stream_cuda.cuh"

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

void central_check_errors(const char* label, STREAM_TYPE* array, STREAM_TYPE avg_err,
                  STREAM_TYPE exp_val, double epsilon, int* errors, ssize_t stream_array_size) {

	if (abs(avg_err/exp_val) > epsilon) {
		(*errors)++;
		printf ("Failed Validation on array %s, AvgRelAbsErr > epsilon (%e)\n", label, epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", exp_val, avg_err, abs(avg_err/exp_val));
	}
}

void standard_errors(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t stream_array_size, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE *aSumErr, STREAM_TYPE *bSumErr, STREAM_TYPE *cSumErr, STREAM_TYPE *aAvgErr, STREAM_TYPE *bAvgErr, STREAM_TYPE *cAvgErr) {
	*aSumErr = 0.0, *bSumErr = 0.0, *cSumErr = 0.0;
	for (ssize_t j = 0; j < stream_array_size; j++) {
		*aSumErr += abs(a[j] - aj);
		*bSumErr += abs(b[j] - bj);
		*cSumErr += abs(c[j] - cj);
	}

	*aAvgErr = *aSumErr / (STREAM_TYPE) stream_array_size;
	*bAvgErr = *bSumErr / (STREAM_TYPE) stream_array_size;
	*cAvgErr = *cSumErr / (STREAM_TYPE) stream_array_size;
}

void central_errors(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t stream_array_size, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, STREAM_TYPE *aSumErr, STREAM_TYPE *bSumErr, STREAM_TYPE *cSumErr, STREAM_TYPE *aAvgErr, STREAM_TYPE *bAvgErr, STREAM_TYPE *cAvgErr) {
	*aSumErr = abs(a[0] - aj);
	*bSumErr = abs(b[0] - bj);
	*cSumErr = abs(c[0] - cj);

	*aAvgErr = *aSumErr;
	*bAvgErr = *bSumErr;
	*cAvgErr = *cSumErr;
}

double validate_values(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t stream_array_size, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c, KernelGroup group) {
	STREAM_TYPE aSumErr, bSumErr, cSumErr;
	STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;

	int err;
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
	
	err = 0;

	switch (group)
	{
	case CENTRAL:
		central_errors(aj, bj, cj, stream_array_size, a, b, c, &aSumErr, &bSumErr, &cSumErr, &aAvgErr, &bAvgErr, &cAvgErr);
		central_check_errors("a[]", a, aAvgErr, aj, epsilon, &err, stream_array_size);
		central_check_errors("b[]", b, bAvgErr, bj, epsilon, &err, stream_array_size);
		central_check_errors("c[]", c, cAvgErr, cj, epsilon, &err, stream_array_size);
		break;
	
	default:
		standard_errors(aj, bj, cj, stream_array_size, a, b, c, &aSumErr, &bSumErr, &cSumErr, &aAvgErr, &bAvgErr, &cAvgErr);
		check_errors("a[]", a, aAvgErr, aj, epsilon, &err, stream_array_size);
		check_errors("b[]", b, bAvgErr, bj, epsilon, &err, stream_array_size);
		check_errors("c[]", c, cAvgErr, cj, epsilon, &err, stream_array_size);
		break;
	}

#ifdef DEBUG
	printf("aSumErr= %f\t\t aAvgErr=%f\n", aSumErr, aAvgErr);
	printf("bSumErr= %f\t\t bAvgErr=%f\n", bSumErr, bAvgErr);
	printf("cSumErr= %f\t\t cAvgErr=%f\n", cSumErr, cAvgErr);
#endif

	return err;
}

void stream_validation(ssize_t stream_array_size, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
	STREAM_TYPE aj,bj,cj;
	int err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;

    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

	/* now execute timing loop  */
	scalar = 3.0;
	for (int k = 0; k < NTIMES; k++){
		// Sequential kernels
		cj = aj;
		bj = scalar * cj;
		cj = aj + bj;
		aj = bj + scalar * cj;
  	}

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c, STREAM);

	if (err == 0) {
		is_validated[COPY] = 1;
		is_validated[SCALE] = 1;
		is_validated[SUM] = 1;
		is_validated[TRIAD] = 1;
	}
#ifdef VERBOSE
	printf("ORIGINAL KERNELS VALIDATION\n");
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

void gather_validation(ssize_t stream_array_size, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
	STREAM_TYPE aj,bj,cj;
	int err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;

    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

	/* now execute timing loop  */
	scalar = 3.0;
	for (int k = 0; k < NTIMES; k++){
		// Gather kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
  	}

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c, GATHER);

	if (err == 0) {
		is_validated[GATHER_COPY] = 1;
		is_validated[GATHER_SCALE] = 1;
		is_validated[GATHER_SUM] = 1;
		is_validated[GATHER_TRIAD] = 1;
	}
#ifdef VERBOSE
	printf("GATHER KERNELS VALIDATION\n");
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

void scatter_validation(ssize_t stream_array_size, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
	STREAM_TYPE aj,bj,cj;
	int err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;

    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

	/* now execute timing loop  */
	scalar = 3.0;
	for (int k = 0; k < NTIMES; k++){
		// Scatter kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
  	}

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c, SCATTER);

	if (err == 0) {
		is_validated[SCATTER_COPY] = 1;
		is_validated[SCATTER_SCALE] = 1;
		is_validated[SCATTER_SUM] = 1;
		is_validated[SCATTER_TRIAD] = 1;
	}
#ifdef VERBOSE
	printf("SCATTER KERNELS VALIDATION\n");
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

void sg_validation(ssize_t stream_array_size, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
	STREAM_TYPE aj,bj,cj;
	int err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;

    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

	/* now execute timing loop  */
	scalar = 3.0;
	for (int k = 0; k < NTIMES; k++){
		// Scatter kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
  	}

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c, SG);

	if (err == 0) {
		is_validated[SG_COPY] = 1;
		is_validated[SG_SCALE] = 1;
		is_validated[SG_SUM] = 1;
		is_validated[SG_TRIAD] = 1;
	}
#ifdef VERBOSE
	printf("SCATTER-GATHER KERNELS VALIDATION\n");
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

void central_validation(ssize_t stream_array_size, STREAM_TYPE scalar, int *is_validated, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
	STREAM_TYPE aj,bj,cj;
	int err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;

    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;

	/* now execute timing loop  */
	scalar = 3.0;
	for (int k = 0; k < NTIMES; k++){
		// Scatter kernels
		cj = aj;
		bj = scalar*cj;
		cj = aj+bj;
		aj = bj+scalar*cj;
  	}

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c, CENTRAL);

	if (err == 0) {
		is_validated[CENTRAL_COPY] = 1;
		is_validated[CENTRAL_SCALE] = 1;
		is_validated[CENTRAL_SUM] = 1;
		is_validated[CENTRAL_TRIAD] = 1;
	}
#ifdef VERBOSE
	printf("SCATTER KERNELS VALIDATION\n");
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

/*--------------------------------------------------------------------------------------
 - Check STREAM results to ensure acuracy
--------------------------------------------------------------------------------------*/

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