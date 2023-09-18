
// void calculateTime(
//     const cudaEvent_t& t0,
//     const cudaEvent_t& t1,
//     double times[NUM_KERNELS][NTIMES],
//     int round,
//     Kernels kernel
// ) {
//     float ms = 0.0;
//     cudaEventSynchronize(t1);
//     cudaEventElapsedTime(&ms, t0, t1);
//     times[kernel][round] = ms * 1E-3;
// }

// void executeSTREAM(
//     STREAM_TYPE* __restrict__ a,
//     STREAM_TYPE* __restrict__ b,
//     STREAM_TYPE* __restrict__ c,
//     STREAM_TYPE* __restrict__ d_a,
//     STREAM_TYPE* __restrict__ d_b,
//     STREAM_TYPE* __restrict__ d_c,
//     ssize_t* __restrict__ d_IDX1,
//     ssize_t* __restrict__ d_IDX2,
//     ssize_t* __restrict__ d_IDX3,
//     double times[NUM_KERNELS][NTIMES],
//     ssize_t stream_array_size,
//     STREAM_TYPE scalar,
//     int is_validated[NUM_KERNELS]
// ) {
//     init_arrays(stream_array_size);
//     cudaEvent_t t0, t1;
//     cudaEventCreate(&t0);
//     cudaEventCreate(&t1);

//     cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

//     for(auto k = 0; k < NTIMES; k++) {
//         cudaEventRecord(t0);
//         stream_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, COPY);

//         cudaEventRecord(t0);
//         stream_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SCALE);

//         cudaEventRecord(t0);
//         stream_add<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SUM);

//         cudaEventRecord(t0);
//         stream_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, TRIAD);
//     }

//     cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

//     stream_validation(stream_array_size, scalar, is_validated, a, b, c);
// }

// void executeGATHER(
//     STREAM_TYPE* __restrict__  a,
//     STREAM_TYPE* __restrict__  b,
//     STREAM_TYPE* __restrict__  c,
//     STREAM_TYPE* __restrict__ d_a,
//     STREAM_TYPE* __restrict__ d_b,
//     STREAM_TYPE* __restrict__ d_c,
//     ssize_t* __restrict__  d_IDX1,
//     ssize_t* __restrict__  d_IDX2,
//     ssize_t* __restrict__  d_IDX3,
//     double times[NUM_KERNELS][NTIMES],
//     ssize_t stream_array_size,
//     STREAM_TYPE scalar,
//     int is_validated[NUM_KERNELS]
// ) {
//     init_arrays(stream_array_size);
//     cudaEvent_t t0, t1;
//     cudaEventCreate(&t0);
//     cudaEventCreate(&t1);

//     cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

//     for(auto k = 0; k < NTIMES; k++) {
//         cudaEventRecord(t0);
//         gather_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, GATHER_COPY);

//         cudaEventRecord(t0);
//         gather_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, GATHER_SCALE);

//         cudaEventRecord(t0);
//         gather_add<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, GATHER_SUM);

//         cudaEventRecord(t0);
//         gather_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, GATHER_TRIAD);
//     }

//     cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

//     gather_validation(stream_array_size, scalar, is_validated, a, b, c);
// }

// void executeSCATTER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
//                     STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
//                     ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
//                     double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
// {
//     init_arrays(stream_array_size);
//     cudaEvent_t t0, t1;
//     cudaEventCreate(&t0);
//     cudaEventCreate(&t1);

//     cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

//     for(auto k = 0; k < NTIMES; k++) {
//         cudaEventRecord(t0);
//         scatter_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SCATTER_COPY);

//         cudaEventRecord(t0);
//         scatter_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SCATTER_SCALE);

//         cudaEventRecord(t0);
//         scatter_add<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SCATTER_SUM);

//         cudaEventRecord(t0);
//         scatter_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SCATTER_TRIAD);
//     }

//     cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

//     scatter_validation(stream_array_size, scalar, is_validated, a, b, c);
// }

// void executeSG(
//     STREAM_TYPE* __restrict__ a,
//     STREAM_TYPE* __restrict__ b,
//     STREAM_TYPE* __restrict__ c,
//     STREAM_TYPE* __restrict__ d_a,
//     STREAM_TYPE* __restrict__ d_b,
//     STREAM_TYPE* __restrict__ d_c,
//     ssize_t* __restrict__ d_IDX1,
//     ssize_t* __restrict__ d_IDX2,
//     ssize_t* __restrict__ d_IDX3,
//     double times[NUM_KERNELS][NTIMES],
//     ssize_t stream_array_size,
//     STREAM_TYPE scalar,
//     int is_validated[NUM_KERNELS]
// ) {
//     init_arrays(stream_array_size);
//     cudaEvent_t t0, t1;
//     cudaEventCreate(&t0);
//     cudaEventCreate(&t1);

//     cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

//     for(auto k = 0; k < NTIMES; k++) {
//         cudaEventRecord(t0);
//         sg_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SG_COPY);

//         cudaEventRecord(t0);
//         sg_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SG_SCALE);

//         cudaEventRecord(t0);
//         sg_add<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SG_SUM);

//         cudaEventRecord(t0);
//         sg_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, SG_TRIAD);
//     }

//     cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

//     sg_validation(stream_array_size, scalar, is_validated, a, b, c);
// }

// void executeCENTRAL(
//     STREAM_TYPE* __restrict__ a,
//     STREAM_TYPE* __restrict__ b,
//     STREAM_TYPE* __restrict__ c,
//     STREAM_TYPE* __restrict__ d_a,
//     STREAM_TYPE* __restrict__ d_b,
//     STREAM_TYPE* __restrict__ d_c,
//     ssize_t* __restrict__  d_IDX1,
//     ssize_t* __restrict__  d_IDX2,
//     ssize_t* __restrict__  d_IDX3,
//     double times[NUM_KERNELS][NTIMES],
//     ssize_t stream_array_size,
//     STREAM_TYPE scalar,
//     int is_validated[NUM_KERNELS]
// ) {
//     init_arrays(stream_array_size);
//     cudaEvent_t t0, t1;
//     cudaEventCreate(&t0);
//     cudaEventCreate(&t1);

//     cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

//     for(auto k = 0; k < NTIMES; k++) {
//         cudaEventRecord(t0);
//         central_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, CENTRAL_COPY);

//         cudaEventRecord(t0);
//         central_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, CENTRAL_SCALE);

//         cudaEventRecord(t0);
//         central_add<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, CENTRAL_SUM);

//         cudaEventRecord(t0);
//         central_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
//         cudaEventRecord(t1);
//         calculateTime(t0, t1, times, k, CENTRAL_TRIAD);
//     }

//     cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

//     central_validation(stream_array_size, scalar, is_validated, a, b, c);
// }