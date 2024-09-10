/* A CUDA hello world! example

We define a Hello workd kernel here that just prints a message and its thread id.
Amount of messages written can be changed by changing the configurations of Thread blocks
in the grid and the amount of thread per block.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void helloKernel(int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d of %d!\n", tid, total);
}

int main() {
    int NTHREADS = 1024;
    cudaError_t cudaStatus;

    helloKernel << <1, NTHREADS >> > (NTHREADS);

    // Wait kernel end its execution
    cudaDeviceSynchronize();

    printf("\n\n Ending execution...\n\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

