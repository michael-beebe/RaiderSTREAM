#ifndef STREAM_CUDA_TUNED_CUH
#define STREAM_CUDA_TUNED_CUH
#include "stream_cuda.cuh"

// =================================================================================
//       				 	  ORIGINAL KERNELS
// =================================================================================
void tuned_STREAM_Copy() {

}

void tuned_STREAM_Scale(STREAM_TYPE scalar) {

}

void tuned_STREAM_Add() {

}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{

}
// =================================================================================
//       				 GATHER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Gather() {

}

void tuned_STREAM_Scale_Gather(STREAM_TYPE scalar) {

}

void tuned_STREAM_Add_Gather() {

}

void tuned_STREAM_Triad_Gather(STREAM_TYPE scalar) {

}

// =================================================================================
//						SCATTER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Scatter() {

}

void tuned_STREAM_Scale_Scatter(STREAM_TYPE scalar) {

}

void tuned_STREAM_Add_Scatter() {

}

void tuned_STREAM_Triad_Scatter(STREAM_TYPE scalar) {

}

// =================================================================================
//					  SCATTER-GATHER VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_SG() {

}

void tuned_STREAM_Scale_SG(STREAM_TYPE scalar) {

}

void tuned_STREAM_Add_SG() {

}

void tuned_STREAM_Triad_SG(STREAM_TYPE scalar) {

}

// =================================================================================
//						CENTRAL VERSIONS OF THE KERNELS
// =================================================================================
void tuned_STREAM_Copy_Central() {

}

void tuned_STREAM_Scale_Central(STREAM_TYPE scalar) {

}

void tuned_STREAM_Add_Central() {

}

void tuned_STREAM_Triad_Central(STREAM_TYPE scalar) {

}

/* end of stubs for the "tuned" versions of the kernels */
#endif
