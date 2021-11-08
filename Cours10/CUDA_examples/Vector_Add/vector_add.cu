#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// CUDA kernel for vector addition
__global__ void addKernel(int* a,int* b, int* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}


// Check vector add result
void verify_result(int N, int *a, int *b, int* c) {
    int correct = 0;
    for (int i = 0; i < N; i++) {
        int sum = a[i] + b[i];
        printf("GPU: %d and CPU %d\n", c[i], sum);
        if (c[i] == sum) correct++;
    }
    printf("%d of %d elements correctly calculated!\n", correct, N);
}

void init_arrays(int* array1, int* array2, int N) {
    for (int i = 0; i < N; i++) {
        array1[i] = rand() % 100;
        array2[i] = rand() % 100;
    }
}

int main() {
    // Array size of 2^16 (65536 elements)
    int array_size = 4096;
    size_t array_bytes = sizeof(int) * array_size;

    // 1) Allocate memory on the device
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, array_bytes);
    cudaMalloc(&d_b, array_bytes);
    cudaMalloc(&d_c, array_bytes);

    // Allocate memory on the host
    int *h_a = (int*) malloc(array_bytes);
    int *h_b = (int*) malloc(array_bytes);
    init_arrays(h_a, h_b, array_size);

    // 2) Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, h_a, array_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, array_bytes, cudaMemcpyHostToDevice);

    // 3) Organize Grid and Block size
    int NUM_THREADS = 256;
    int NUM_BLOCKS = array_size / NUM_THREADS;
    //int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // 3.1) Call kernel
    addKernel <<<NUM_BLOCKS, NUM_THREADS>>> (d_a, d_b, d_c, array_size);

    // 4) Copy result GPU to CPU
    int* h_c = (int*)malloc(array_bytes);
    cudaMemcpy(h_c, d_c, array_bytes, cudaMemcpyDeviceToHost);

    // 5) Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Check result for errors
    verify_result(array_size, h_a, h_b, h_c);
    return 0;
}
