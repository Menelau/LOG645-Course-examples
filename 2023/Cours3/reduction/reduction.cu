/* SUM reduction kernel example inspired by NVIDIA by Mark harris tutorial:
* https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf  
*
*/
#include "cuda_runtime.h""
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define SIZE 256

__global__ void sumReductionWarpDisagreement(unsigned int* v, unsigned int* v_r) {

	// Allocate shared memory
	__shared__ unsigned int partial_sum[SIZE];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	// Wait for all threads to load into shared memory
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if (threadIdx.x % (2 * stride) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
		}
		__syncthreads();
	}

	// Let the thread 0 write it's result to main memory (visible to other threads)
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}


__global__ void sumReductionWarpAgreement(unsigned int* v, unsigned int* v_r) {
	
	// Use shared memory to share data in a block
	__shared__ unsigned int partial_sum[SIZE];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = v[tid];
	// Wait for all threads to load into shared memory
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < stride) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) 
		v_r[blockIdx.x] = partial_sum[0];
	
}


int main(void) {

	unsigned int N = 65536;			// this is the maximum value this implementation allows.
	size_t bytes = N * sizeof(unsigned int);

	// Original vector and result vector
	unsigned int* vector, * vector_reduced;

	cudaMallocManaged(&vector, bytes);
	cudaMallocManaged(&vector_reduced, bytes);

	for (int i = 0; i < N; i++) {
		vector[i] = 1;
	}
	
	int BLOCK_SIZE = SIZE;
	unsigned int GRID_SIZE = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
	// Call kernel
	sumReductionWarpAgreement <<<GRID_SIZE, BLOCK_SIZE >>> (vector, vector_reduced);
	sumReductionWarpAgreement <<<1, BLOCK_SIZE >>> (vector_reduced, vector_reduced);
	cudaDeviceSynchronize();

	printf("Result is %u \n", vector_reduced[0]);

	return 0;
}
