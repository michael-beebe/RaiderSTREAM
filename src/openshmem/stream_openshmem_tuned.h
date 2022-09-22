#ifndef STREAM_OPENSHMEM_TUNED_H
#define STREAM_OPENSHMEM_TUNED_H

#include "stream_openshmem.h"

/* stubs for "tuned" versions of the kernels */
#ifdef TUNED
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

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_ScatterGather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[c_idx[j]] = a[a_idx[j]];
}

void tuned_STREAM_Scale_ScatterGather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[b_idx[j]] = scalar * c[c_idx[j]];
}

void tuned_STREAM_Add_ScatterGather() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[a_idx[j]] = a[a_idx[j]] + b[b_idx[j]];
}

void tuned_STREAM_Triad_ScatterGather(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[a_idx[j]] = b[b_idx[j]] + scalar * c[c_idx[j]];
}

// =================================================================================
//						CENTRAL VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Central() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[0] = a[0];
}

void tuned_STREAM_Scale_Central(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		b[0] = scalar * c[0];
}

void tuned_STREAM_Add_Central() {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		c[0] = a[0] + b[0];
}

void tuned_STREAM_Triad_Central(STREAM_TYPE scalar) {
	ssize_t j;
#pragma omp parallel for
	for (j=0; j<stream_array_size; j++)
		a[0] = b[0] + scalar * c[0];
}
/* end of stubs for the "tuned" versions of the kernels */
#endif // TUNED

#endif // STREAM_OPENSHMEM_TUNED_H