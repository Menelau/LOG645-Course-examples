#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Kernel to perform a simple matrix multiplication. We will optimize it later...
__global__ void matrixMultiplicationKernel(int *matrix_A, int *matrix_B, int *matrix_C, int N) {
  // Compute each thread's x and y indexes
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Iterate over row, and column
  matrix_C[row * N + col] = 0;
  for (int i = 0; i < N; i++) {
    // Accumulate results for a single element in the output matrix
    matrix_C[row * N + col] += matrix_A[row * N + i] * matrix_B[i * N + col];
  }
}

// Functon to compare the GPU results with the calculation performed in the CPU.
int verifyResults(int *matrix_A, int *matrix_B, int *matrix_C, int N) {
    int correct = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;     //not a perfect nested loop :-(
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += matrix_A[i * N + k] * matrix_B[k * N + j];
            }
            // Check against the CPU result
            if (tmp == matrix_C[i * N + j]) correct++;
        }
    }
    printf("%d of %d values verified!", N, N);
    return 0;
 }


int main() {

  // Defining a square matrix of size N X N
  int N = 2048;
  size_t matrix_size = N * N * sizeof(int);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs to get a 2D grid and 2D ThreadBlock
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Host vectors
  int *h_A = (int*) malloc(matrix_size);
  int *h_B = (int*) malloc(matrix_size);
  int *h_C = (int*) malloc(matrix_size);
  int *d_A, *d_B, *d_C;

  // Initialize matrices A and B in the CPU
  for (int i = 0; i < N*N; i++){
    h_A[i] = rand() % 100;
    h_B[i] = rand() % 100;
  }

  // 1) Allocate CUDA memory
  cudaMalloc(&d_A, matrix_size);
  cudaMalloc(&d_B, matrix_size);
  cudaMalloc(&d_C, matrix_size);

  // 2) Copy data from CPU to the device
  cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

  // 3) Launch kernel
  matrixMultiplicationKernel <<<blocks, threads>>>(d_A, d_B, d_C, N);

  // 4) Copy back to the host. This function waits for the kernel (synchronous)
  cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);

  // Check result
  verifyResults(h_A, h_B, h_C, N);

  // 5) Free memory on device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

