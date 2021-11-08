
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


// CUDA kernel that squares each number in a vector.
__global__ void squareKernel(int* input, int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = input[tid] * input[tid];
}

// Sequential version to guarantee thatthe results GPU - CPU are the same
int verifyResult(int* output, int* input, int N) {
    for (int i = 0; i < N; i++) {
        printf("GPU: %d and CPU: %d\n", output[i], input[i]*input[i]);
    }
    printf("%d of %d array elements verified with success!\n", N, N);
    return 0;
}

// Initialize Array with N integer numbers from 0 to 99
void initializeArray(int* array, int N) {
    for (int i = 0; i < N; i++)
        array[i] = rand() % 100;
}

int main() {

    const int array_size = 2048;

    // Size of the data that will be transfered between CPU - GPU
    size_t bytes = sizeof(int) * array_size;

    // Pointers holding the CPU data
    int* h_input, * h_output;
    h_input = (int*)malloc(bytes);
    h_output = (int*)malloc(bytes);

    //get random values for the input array
    initializeArray(h_input, array_size);

    // pointers holding the GPU data
    int* d_input, * d_output;

    // Variable o collect errors in cuda function calls
    cudaError_t cudaStatus;

    // Optional - Choose which GPU to run on, needed in a Multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**) &d_input, bytes);
    cudaMalloc((void**) &d_output, bytes);

    // Copy input vectors from host memory to GPU buffers
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element
    squareKernel <<<8, 256>>> (d_input, d_output);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation and acts as a synchronization
    // barrier. No need to call cudaDeviceSynchronize() for that
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    int ver = verifyResult(h_output, h_input, array_size);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
