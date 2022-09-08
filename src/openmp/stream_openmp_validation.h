#ifndef STREAM_OPENMP_VALIDATION_H
#define STREAM_OPENMP_VALIDATION_H

#include "stream_openmp.h"

double validate_values(STREAM_TYPE aj, STREAM_TYPE bj, STREAM_TYPE cj, ssize_t stream_array_size, STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c) {
	STREAM_TYPE aSumErr, bSumErr, cSumErr;
	STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;

	int err;

	double epsilon;

	/* accumulate deltas between observed and expected results */
	aSumErr = 0.0, bSumErr = 0.0, cSumErr = 0.0;
	for (ssize_t j = 0; j < stream_array_size; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}

	aAvgErr = aSumErr / (STREAM_TYPE) stream_array_size;
	bAvgErr = bSumErr / (STREAM_TYPE) stream_array_size;
	cAvgErr = cSumErr / (STREAM_TYPE) stream_array_size;

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

#ifdef DEBUG
	printf("aSumErr= %f\t\t aAvgErr=%f\n", aSumErr, aAvgErr);
	printf("bSumErr= %f\t\t bAvgErr=%f\n", bSumErr, bAvgErr);
	printf("cSumErr= %f\t\t cAvgErr=%f\n", cSumErr, cAvgErr);
#endif

	// Check errors on each array
	check_errors("a[]", a, aAvgErr, aj, epsilon, &err, stream_array_size);
	check_errors("b[]", b, bAvgErr, bj, epsilon, &err, stream_array_size);
	check_errors("c[]", c, cAvgErr, cj, epsilon, &err, stream_array_size);

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

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c);

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

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c);

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

	err = validate_values(aj, bj, cj, stream_array_size, a, b, c);

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

#endif