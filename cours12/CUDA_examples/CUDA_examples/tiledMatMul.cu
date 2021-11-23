#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//static shared memory allocation.
#define TILE_WIDTH 16


__global__ void matMulTiledKernel(float* M, float* N, float* P, int Width) {
    // static shared memory allocation
    __shared__ float shared_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_N[TILE_WIDTH][TILE_WIDTH];

    //keeping as we gonna use later
    int tx = threadIdx.x;    int ty = threadIdx.y;
    // Identify the row and column of the P element to work on
    int Row = blockIdx.y * blockDim.y + ty;
    int Col = blockIdx.x * blockDim.x + tx;
    float Pvalue = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int i = 0; i < Width / TILE_WIDTH; i++) {

        // Collaborative loading of M and N tiles into shared memory
        shared_M[ty][tx] = M[(Row * Width) + (i * TILE_WIDTH) + tx];
        shared_N[ty][tx] = N[Col + (i * TILE_WIDTH + ty) * Width];
        // synchronization barrier to guarantee all data is ready.
        __syncthreads();
        // Computation using shared memory
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += shared_M[ty][k] * shared_N[k][tx];
        // Synchronization barrier to guarantee the whole tile is computed before
        // moving to another tile.
        __syncthreads();
    }
    P[(Row * Width) + Col] = Pvalue;
}


int verifyResults(float* matrix_A, float* matrix_B, float* matrix_C, int N);


int main(void) {

    int N = 1024;
    int n_elements = N * N;
    float* A, * B, * C;
    size_t bytes = sizeof(float) * n_elements;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    for (int i = 0; i < N * N; i++) {

        A[i] = rand() % 100;
        B[i] = rand() % 100;
        C[i] = -1;
    }

    int BLOCK_SIZE_X = 16;
    int BLOCK_SIZE_Y = 16;
    int GRID_SIZE_X = (N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int GRID_SIZE_Y = (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;

    dim3 BLOCK_DIM(BLOCK_SIZE_X, BLOCK_SIZE_X);
    dim3 GRID_DIM(GRID_SIZE_X, GRID_SIZE_Y);

    matMulTiledKernel <<<GRID_DIM, BLOCK_DIM>>> (A, B, C, N);
    cudaDeviceSynchronize();
    printf("Finished GPU calculation\n");

    //validate results here.
    verifyResults(A, B, C, N);
	return 0;
}


// Functon to compare the GPU results with the calculation performed in the CPU.
int verifyResults(float* matrix_A, float* matrix_B, float* matrix_C, int N) {
    long correct = 0;
    #pragma omp parallel for collapse(3)
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
    printf("%ld of %ld values verified!\n", correct, (long)N * N);
    return 0;
}
