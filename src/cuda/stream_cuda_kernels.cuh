



__global__ void stream_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
	ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[j];
}

__global__ void stream_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[j] = scalar * d_c[j];
}

__global__ void stream_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[j] + d_b[j];
}

__global__ void stream_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[j] = d_b[j] + scalar * d_c[j];
}

void calculateTime(const cudaEvent_t& t0, const cudaEvent_t& t1, double times[NUM_KERNELS][NTIMES], int round, Kernels kernel) {
	float ms = 0.0;
	cudaEventSynchronize(t1);
	cudaEventElapsedTime(&ms, t0, t1);
	times[kernel][round] = ms * 1E-3;
}

void executeSTREAM(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		stream_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, COPY);

		cudaEventRecord(t0);
		stream_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCALE);

		cudaEventRecord(t0);
		stream_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SUM);

		cudaEventRecord(t0);
		stream_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	stream_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void gather_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[d_IDX1[j]];
}

__global__ void gather_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[j] = scalar * d_c[d_IDX2[j]];
}

__global__ void gather_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[j] = d_a[d_IDX1[j]] + d_b[d_IDX2[j]];
}

__global__ void gather_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[j] = d_b[d_IDX1[j]] + scalar * d_c[d_IDX2[j]];
}

void executeGATHER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		gather_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_COPY);

		cudaEventRecord(t0);
		gather_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_SCALE);

		cudaEventRecord(t0);
		gather_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_SUM);

		cudaEventRecord(t0);
		gather_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, GATHER_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	gather_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void scatter_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[j];
}

__global__ void scatter_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[d_IDX2[j]] = scalar * d_c[j];
}

__global__ void scatter_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[j] + d_b[j];
}

__global__ void scatter_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[d_IDX2[j]] = d_b[j] + scalar * d_c[j];
}

void executeSCATTER(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		scatter_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_COPY);

		cudaEventRecord(t0);
		scatter_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_SCALE);

		cudaEventRecord(t0);
		scatter_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_SUM);

		cudaEventRecord(t0);
		scatter_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SCATTER_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	scatter_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void sg_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[d_IDX2[j]];
}

__global__ void sg_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_b[d_IDX2[j]] = scalar * d_c[d_IDX1[j]];
}

__global__ void sg_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_c[d_IDX1[j]] = d_a[d_IDX2[j]] + d_b[d_IDX3[j]];
}

__global__ void sg_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t* __restrict__ d_IDX1, ssize_t* __restrict__ d_IDX2, ssize_t* __restrict__ d_IDX3, ssize_t stream_array_size) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < stream_array_size) d_a[d_IDX2[j]] = d_b[d_IDX3[j]] + scalar * d_c[d_IDX1[j]];
}

void executeSG(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		sg_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_COPY);

		cudaEventRecord(t0);
		sg_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_SCALE);

		cudaEventRecord(t0);
		sg_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_SUM);

		cudaEventRecord(t0);
		sg_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, d_IDX1, d_IDX2, d_IDX3, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, SG_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	sg_validation(stream_array_size, scalar, is_validated, a, b, c);
}

__global__ void central_copy(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
	d_c[0] = d_a[0];
}

__global__ void central_scale(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
	d_b[0] = scalar * d_c[0];
}

__global__ void central_sum(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, ssize_t stream_array_size) {
	d_c[0] = d_a[0] + d_b[0];
}

__global__ void central_triad(STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c, STREAM_TYPE scalar, ssize_t stream_array_size) {
	d_a[0] = d_b[0] + scalar * d_c[0];
}

void executeCENTRAL(STREAM_TYPE* __restrict__   a, STREAM_TYPE* __restrict__   b, STREAM_TYPE* __restrict__  c,
				   STREAM_TYPE* __restrict__ d_a, STREAM_TYPE* __restrict__ d_b, STREAM_TYPE* __restrict__ d_c,
				   ssize_t* __restrict__  d_IDX1, ssize_t* __restrict__  d_IDX2, ssize_t* __restrict__  d_IDX3,
				   double times[NUM_KERNELS][NTIMES], ssize_t stream_array_size, STREAM_TYPE scalar, int is_validated[NUM_KERNELS])
{
	init_arrays(stream_array_size);
	cudaEvent_t t0, t1;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);

	cudaMemcpy(d_a, a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyHostToDevice);

	for(auto k = 0; k < NTIMES; k++) {
		cudaEventRecord(t0);
		central_copy<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_COPY);

		cudaEventRecord(t0);
		central_scale<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_SCALE);

		cudaEventRecord(t0);
		central_sum<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_SUM);

		cudaEventRecord(t0);
		central_triad<<< (stream_array_size + 255)/256, 256 >>>(d_a, d_b, d_c, scalar, stream_array_size);
		cudaEventRecord(t1);
		calculateTime(t0, t1, times, k, CENTRAL_TRIAD);
	}

	cudaMemcpy(a, d_a, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeof(STREAM_TYPE) * stream_array_size, cudaMemcpyDeviceToHost);

	central_validation(stream_array_size, scalar, is_validated, a, b, c);
}