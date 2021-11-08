#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h> 

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
            prop.major, prop.minor);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
            prop.multiProcessorCount,
            _ConvertSMVer2Cores(prop.major, prop.minor),
            _ConvertSMVer2Cores(prop.major, prop.minor) *
           prop.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f ""GHz)\n",
            prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
        printf("  Total amount of constant memory:               %lu bytes\n", prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", prop.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
            prop.maxThreadsDim[2]); 
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
            prop.maxGridSize[2]);
        printf("  Maximum memory pitch %lu bytes\n", prop.memPitch);
    }
}