/* Code to demonstrate the use of shared mermory in a static and dynamic context
 *
 * Modified from the examples provided by NVIDIA
 */

 /* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions
  * are met:
  *  * Redistributions of source code must retain the above copyright
  *    notice, this list of conditions and the following disclaimer.
  *  * Redistributions in binary form must reproduce the above copyright
  *    notice, this list of conditions and the following disclaimer in the
  *    documentation and/or other materials provided with the distribution.
  *  * Neither the name of NVIDIA CORPORATION nor the names of its
  *    contributors may be used to endorse or promote products derived
  *    from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define SIZE 256
__global__ void staticReverse(int* d, int *result, int N) {
    __shared__ int s[256];
    int t = threadIdx.x;
    int tr = N - t - 1;
    s[t] = d[t];
    // Barrier to guarantee all threads copied data to the shared memory
    __syncthreads();

    result[t] = s[tr];
}

__global__ void dynamicReverse(int* d, int* result, int N) {
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = N - t - 1;
    s[t] = d[t];

    // Barrier to guarantee all threads copied data to the shared memory
    __syncthreads();
    result[t] = s[tr];
}

int main(void) {

    const int N = 256;
    int *vec, * result, *groundTruth;

    cudaMallocManaged(&vec, N * sizeof(int));
    cudaMallocManaged(&result, N * sizeof(int));
    cudaMallocManaged(&groundTruth, N * sizeof(int));

    // initializing arrays
    for (int i = 0; i < N; i++) {
        vec[i] = i;
        result[i] = 0;
        groundTruth[i] = N - i - 1;
    }

    // run static shared memory version
    staticReverse <<<1, N >>> (vec, result, N);
    
    // Need a synchronization barrier here since the kernel call is Asynchronous
    // and we don't have a call to cudaMemcpy that imposes a barrier.
    cudaDeviceSynchronize();

    // check if the results are consistent.
    for (int i = 0; i < N; i++){
        if (result[i] != groundTruth[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, result[i], groundTruth[i]);
            return -1;
        }
    }

    // run dynamic shared memory version
    dynamicReverse<<<1, N, N * sizeof(int)>>> (vec, result, N);
    cudaDeviceSynchronize();

    // Check if the result is consistent
    for (int i = 0; i < N; i++) {
        if (result[i] != groundTruth[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, result[i], groundTruth[i]);
            return -1;
        }
    }
    cudaFree(vec);
    cudaFree(groundTruth);
    cudaFree(result);
    printf("No errors!");
    return 0;
}
