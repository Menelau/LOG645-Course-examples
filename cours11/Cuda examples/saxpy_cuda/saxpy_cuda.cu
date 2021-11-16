
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__ void saxpyKernel(float alpha, const float* A, const float* B, float* C, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
        C[tid] = (alpha * A[tid]) + B[tid];
}

// saxpy CPU implementation
void verify_results(float alpha, float* A, const float* B, float* C, int N) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (C[i] == (alpha * A[i]) + B[i]) count++;
    }
    printf("%d of %d values correctly verified!\n", count, N);
}

int main(int argc, char** argv) {

    int N = (1 << 10) + 7;
    float alpha = 0.5;
    float* A, * B, * C;
    size_t array_bytes = N * sizeof(float);

    // Using unified memory to make things EVEN EASIER!
    cudaMallocManaged(&A, array_bytes);
    cudaMallocManaged(&B, array_bytes);
    cudaMallocManaged(&C, array_bytes);

    //initialize arrays with a random numbers
    for (int i = 0; i < N; i++) {
        A[i] = ((float)(rand() % 100)) / 100.0f;
        B[i] = ((float)(rand() % 100)) / 100.0f;
        C[i] = -1;
    }
    
    // Threads and blocks configuration for a non 
    int n_threads = 256;
    int n_blocks = (N + n_threads - 1) / n_threads;
    printf("%d blocks and %d threads per block for a total %d threads\n", n_blocks, n_threads, n_blocks * n_threads);

    // call kernel. Nothing changes here.
    saxpyKernel<<<n_blocks, n_threads>>> (alpha, A, B, C, N);

    // Need a synchronization barrier here since the kernel call is Asynchronous 
    // with respect to the host code and we don't have a call to cudaMemcpy that imposes a barrier.
    cudaDeviceSynchronize();

    // Display results
    for (int i = 0; i < N; i++) {
        printf("index %d: %f * %f + %f = %f\n", i, alpha, A[i], B[i], C[i]);
    }
    //verify_results(alpha, A, B, C, N);

    cudaFree(A);
    cudaFree(A);
    cudaFree(B);

}

