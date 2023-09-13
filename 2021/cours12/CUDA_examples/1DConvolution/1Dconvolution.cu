#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASK_LENGTH 9
#define BLOCK_DIM 256
#define EPSILON 0.0001

// Allocate mask in the constant memory so we can leverage its cache
__constant__ int d_mask[MASK_LENGTH];


// Kernel with a naive 1D convolution implementation
__global__ void basic1Dconv(float* N, float* P, float* mask, int Mask_Width, int Width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N_start_point = i - (Mask_Width / 2);
    float Pvalue = 0.0;

    for (int j = 0; j < Mask_Width; j++) {
        // check for out of bounds (ghost values = 0)
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * mask[j];
        }
    }
    P[i] = Pvalue;
}


// kernel using the constant memory for reducing half of the global memory access.
__global__ void constantMemory1Dconv(float* N, float* P, int Width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (MASK_LENGTH / 2);
    for (int j = 0; j < MASK_LENGTH; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * d_mask[j];
        }
    }
    P[i] = Pvalue;
}


// kernel using the tiled + constant memory for larger improvement.
__global__ void tiledConvolution1DKernel(float* N, float* P, int Width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_N[BLOCK_DIM + MASK_LENGTH - 1];
    int n = MASK_LENGTH / 2;

    //copy halo left
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - n) {
        shared_N[threadIdx.x - (blockDim.x - n)] =
            (halo_index_left < 0) ? 0 : N[halo_index_left];
    }
    // copy mid elements
    shared_N[n + threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];

    //copy halo right
    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) {
        shared_N[n + blockDim.x + threadIdx.x] =
            (halo_index_right >= Width) ? 0 : N[halo_index_right];
    }

    __syncthreads();
    
    float Pvalue = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
        Pvalue += shared_N[threadIdx.x + j] * d_mask[j];
    }
    P[tid] = Pvalue;
}


void verify_result(float* array, float* result, float* mask, int N);


int main() {
    // initializing the random seed
    srand(time(NULL));

    int N = 1 << 25;

    int bytes_array = N * sizeof(float);
    int bytes_mask = MASK_LENGTH * sizeof(float);

    float* vec, * conv_mask, *result;
    cudaMallocManaged(&vec, bytes_array);
    cudaMallocManaged(&result, bytes_array);
    cudaMallocManaged(&conv_mask, bytes_mask);
    
    // initialize vector with random float numbers
    for (int i = 0; i < N; i++) {
        vec[i] = (float)rand() / (float)RAND_MAX;
        result[i] = -1;
    }
    
    // initialize convolution mask with random float numbers
    for (int i = 0; i < MASK_LENGTH; i++) {
        conv_mask[i] = (float)rand() / (float)RAND_MAX;
    }

    //copy mask to the constant
    cudaMemcpyToSymbol(d_mask, conv_mask, bytes_mask);

    // 1D convolution so using a 1D grid here.
    int GRID_DIM = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    // Call simple kernel
    //basic1Dconv <<<GRID_DIM, BLOCK_DIM >> > (vec, result, conv_mask, MASK_LENGTH, N);
    //cudaDeviceSynchronize();
    //printf("Finished GPU simple kernel\n");

    //-----------------------------------------------------------------------------------------
    
    // Call simple kernel
    //constantMemory1Dconv<<<GRID_DIM, BLOCK_DIM>>> (vec, result, N);
    //cudaDeviceSynchronize();
    printf("Finished GPU constant memory kernel\n");

    //--------------------------------------------------------------
    // 
    
    // Call constant and shared memory kernel
    tiledConvolution1DKernel<<<GRID_DIM, BLOCK_DIM>>> (vec, result, N);
    cudaDeviceSynchronize();
    printf("Finished GPU shared and constant memory kernel\n");

    verify_result(vec, result, conv_mask, N);

    cudaFree(result);
    cudaFree(vec);
    cudaFree(conv_mask);

    return 0;
}

// Compare results with CPU implementation.
void verify_result(float* array, float* result, float* mask, int N) {
    int correct = 0;
    int radius = MASK_LENGTH / 2;

    float temp;
    int start;
    
    for (int i = 0; i < N; i++) {
        start = i - radius;
        temp = 0.0;
        for (int j = 0; j < MASK_LENGTH; j++) {
            if ((start + j >= 0) && (start + j < N)) {
                temp += array[start + j] * mask[j];
            }
        }
        if ((temp - result[i]) < EPSILON) 
            correct++;

    }
    printf("%ld of %ld values verified!\n", correct, N);
}
