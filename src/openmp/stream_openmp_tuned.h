#ifndef STREAM_OPENMP_TUNED_H
#define STREAM_OPENMP_TUNED_H

#include "stream_openmp.h"

#ifdef TUNED
void tuned_STREAM_Copy();
void tuned_STREAM_Scale(STREAM_TYPE scalar);
void tuned_STREAM_Add();
void tuned_STREAM_Triad(STREAM_TYPE scalar);
void tuned_STREAM_Copy_Gather();
void tuned_STREAM_Scale_Gather(STREAM_TYPE scalar);
void tuned_STREAM_Add_Gather();
void tuned_STREAM_Triad_Gather(STREAM_TYPE scalar);
void tuned_STREAM_Copy_Scatter();
void tuned_STREAM_Scale_Scatter(STREAM_TYPE scalar);
void tuned_STREAM_Add_Scatter();
void tuned_STREAM_Triad_Scatter(STREAM_TYPE scalar);
#endif // TUNED

/* stubs for "tuned" versions of the kernels */
#ifdef TUNED

ssize_t stream_array_size;
parse_opts(argc, argv, &stream_array_size);

// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
void tuned_STREAM_Copy()
{
	ssize_t j;
#pragma omp parallel for
    for (j=0; j<stream_array_size; j++)
        c[j] = a[j];
}

void tuned_STREAM_Scale(STREAM_TYPE scalar)
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    b[j] = scalar*c[j];
}

void tuned_STREAM_Add()
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
	    a[j] = b[j]+scalar*c[j];
}
// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Gather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[j] = a[a_idx[j]];
}

void tuned_STREAM_Scale_Gather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[j] = scalar * c[c_idx[j]];
}

void tuned_STREAM_Add_Gather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[j] = a[a_idx[j]] + b[b_idx[j]];
}

void tuned_STREAM_Triad_Gather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[j] = b[b_idx[j]] + scalar * c[c_idx[j]];
}

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Scatter() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[c_idx[j]] = a[j];
}

void tuned_STREAM_Scale_Scatter(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[b_idx[j]] = scalar * c[j];
}

void tuned_STREAM_Add_Scatter() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[a_idx[j]] = a[j] + b[j];
}

void tuned_STREAM_Triad_Scatter(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[a_idx[j]] = b[j] + scalar * c[j];
}
/* end of stubs for the "tuned" versions of the kernels */
#endif

#endif // STREAM_OPENMP_TUNED_H