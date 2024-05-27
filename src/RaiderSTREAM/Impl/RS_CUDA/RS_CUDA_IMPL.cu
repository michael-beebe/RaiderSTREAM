/**************************************************
* @brief Copies data from one device array to another.
* 
* @param d_a Source device array.
* @param d_b Unused in this function.
* @param d_c Destination device array.
* @param streamArraySize Size of the stream array.
**************************************************/
 __global__ void seqCopy(
    STREAM_TYPE* __restrict__ d_a,
    STREAM_TYPE* __restrict__ d_b,
    STREAM_TYPE* __restrict__ d_c,
    ssize_t streamArraySize
  ) {
    ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < streamArraySize) d_c[j] = d_a[j];
  }
  
/**************************************************
* @brief Scales data in a device array.
* 
* @param d_a Unused in this function.
* @param d_b Destination device array.
* @param d_c Source device array.
* @param scalar Scalar value for scaling.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void seqScale(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t streamArraySize)
{ 
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_b[j] = scalar * d_c[j];
}

/**************************************************
* @brief Adds data from two device arrays and stores in a third.
* 
* @param d_a First source device array.
* @param d_b Second source device array.
* @param d_c Destination device array.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void seqAdd(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < streamArraySize) d_c[j] = d_a[j] + d_b[j];
}
  
/**************************************************
* @brief Performs triad operation on device arrays.
* 
* @param d_a Destination device array.
* @param d_b Source device array.
* @param d_c Source device array.
* @param scalar Scalar value for operations.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void seqTriad(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_a[j] = d_b[j] + scalar * d_c[j];
}  

/**************************************************
* @brief Copies data using gather operation.
* 
* @param d_a Source device array.
* @param d_b Unused in this function.
* @param d_c Destination device array.
* @param d_IDX1 Index array for gather operation.
* @param d_IDX2 Unused in this function.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void gatherCopy(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_c[j] = d_a[d_IDX1[j]];
}
  
/**************************************************
* @brief Scales data using gather operation.
* 
* @param d_a Unused in this function.
* @param d_b Destination device array.
* @param d_c Source device array.
* @param scalar Scalar value for scaling.
* @param d_IDX1 Unused in this function.
* @param d_IDX2 Index array for gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void gatherScale(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_b[j] = scalar * d_c[d_IDX2[j]];
}
  
/**************************************************
* @brief Adds data using gather operation.
* 
* @param d_a First source device array.
* @param d_b Second source device array.
* @param d_c Destination device array.
* @param d_IDX1 First index array for gather operation.
* @param d_IDX2 Second index array for gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void gatherAdd(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_c[j] = d_a[d_IDX1[j]] + d_b[d_IDX2[j]];
}

/**************************************************
* @brief Performs triad operation using gather.
* 
* @param d_a Destination device array.
* @param d_b Source device array.
* @param d_c Source device array.
* @param scalar Scalar value for operations.
* @param d_IDX1 First index array for gather operation.
* @param d_IDX2 Second index array for gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void gatherTriad(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_a[j] = d_b[d_IDX1[j]] + scalar * d_c[d_IDX2[j]];
}

/**************************************************
* @brief Copies data from one device array to another using scatter operation.
* 
* @param d_a Source device array.
* @param d_b Unused in this function.
* @param d_c Destination device array.
* @param d_IDX1 Index array for scatter operation.
* @param d_IDX2 Unused in this function.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void scatterCopy(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize
) {
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_c[d_IDX1[j]] = d_a[j];
}

/**************************************************
* @brief Scales data in a device array using scatter operation.
* 
* @param d_a Unused in this function.
* @param d_b Destination device array.
* @param d_c Source device array.
* @param scalar Scalar value for scaling.
* @param d_IDX1 Unused in this function.
* @param d_IDX2 Index array for scatter operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void scatterScale(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize
) {
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_b[d_IDX2[j]] = scalar * d_c[j];
}

/**************************************************
* @brief Adds data from two device arrays and stores in a third using scatter operation.
* 
* @param d_a First source device array.
* @param d_b Second source device array.
* @param d_c Destination device array.
* @param d_IDX1 First index array for scatter operation.
* @param d_IDX2 Second index array for scatter operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void scatterAdd(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize
) {
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_c[d_IDX1[j]] = d_a[j] + d_b[j];
}

/**************************************************
* @brief Performs triad operation on device arrays using scatter operation.
* 
* @param d_a Destination device array.
* @param d_b Source device array.
* @param d_c Source device array.
* @param scalar Scalar value for operations.
* @param d_IDX1 First index array for scatter operation.
* @param d_IDX2 Second index array for scatter operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void scatterTriad(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t streamArraySize
) {
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_a[d_IDX2[j]] = d_b[j] + scalar * d_c[j];
}

/**************************************************
* @brief Copies data using scatter-gather operation.
* 
* @param d_a Source device array.
* @param d_b Unused in this function.
* @param d_c Destination device array.
* @param d_IDX1 First index array for scatter-gather operation.
* @param d_IDX2 Second index array for scatter-gather operation.
* @param d_IDX3 Third index array for scatter-gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void sgCopy(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t* __restrict__ d_IDX3,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_c[d_IDX1[j]] = d_a[d_IDX2[j]];
}
  
/**************************************************
* @brief Scales data using scatter-gather operation.
* 
* @param d_a Unused in this function.
* @param d_b Destination device array.
* @param d_c Source device array.
* @param scalar Scalar value for scaling.
* @param d_IDX1 First index array for scatter-gather operation.
* @param d_IDX2 Second index array for scatter-gather operation.
* @param d_IDX3 Third index array for scatter-gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void sgScale(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t* __restrict__ d_IDX3,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_b[d_IDX2[j]] = scalar * d_c[d_IDX1[j]];
}
  
/**************************************************
* @brief Adds data using scatter-gather operation.
* 
* @param d_a First source device array.
* @param d_b Second source device array.
* @param d_c Destination device array.
* @param d_IDX1 First index array for scatter-gather operation.
* @param d_IDX2 Second index array for scatter-gather operation.
* @param d_IDX3 Third index array for scatter-gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void sgAdd(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t* __restrict__ d_IDX3,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_c[d_IDX1[j]] = d_a[d_IDX2[j]] + d_b[d_IDX3[j]];
}
  
/**************************************************
* @brief Performs triad operation using scatter-gather.
*
* @param d_a Destination device array.
* @param d_b Source device array.
* @param d_c Source device array.
* @param scalar Scalar value for operations.
* @param d_IDX1 First index array for scatter-gather operation.
* @param d_IDX2 Second index array for scatter-gather operation.
* @param d_IDX3 Third index array for scatter-gather operation.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void sgTriad(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t* __restrict__ d_IDX1,
  ssize_t* __restrict__ d_IDX2,
  ssize_t* __restrict__ d_IDX3,
  ssize_t streamArraySize)
{
  ssize_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < streamArraySize) d_a[d_IDX2[j]] = d_b[d_IDX3[j]] + scalar * d_c[d_IDX1[j]];
}
  
/**************************************************
* @brief Copies data to a central location.
* 
* @param d_a Source device array.
* @param d_b Unused in this function.
* @param d_c Destination device array.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void centralCopy(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t streamArraySize)
{
  d_c[0] = d_a[0];
}
  
/**************************************************
* @brief Scales data at a central location.
* 
* @param d_a Unused in this function.
* @param d_b Destination device array.
* @param d_c Source device array.
* @param scalar Scalar value for scaling.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void centralScale(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t streamArraySize)
{
  d_b[0] = scalar * d_c[0];
}
  
/**************************************************
* @brief Adds data at a central location.
* 
* @param d_a First source device array.
* @param d_b Second source device array.
* @param d_c Destination device array.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void centralAdd(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  ssize_t streamArraySize)
{
  d_c[0] = d_a[0] + d_b[0];
}
  
/**************************************************
* @brief Performs triad operation at a central location.
* 
* @param d_a Destination device array.
* @param d_b Source device array.
* @param d_c Source device array.
* @param scalar Scalar value for operations.
* @param streamArraySize Size of the stream array.
**************************************************/
__global__ void centralTriad(
  STREAM_TYPE* __restrict__ d_a,
  STREAM_TYPE* __restrict__ d_b,
  STREAM_TYPE* __restrict__ d_c,
  STREAM_TYPE scalar,
  ssize_t streamArraySize)
{
  d_a[0] = d_b[0] + scalar * d_c[0];
}


/* EOF */