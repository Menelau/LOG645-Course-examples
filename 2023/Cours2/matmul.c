/* Code example to demonstrate the effect of good cache access and SIMD
parallelisation. For simplicity, we are considering that both matrices are
squared.

*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Naive matrix multiplication function
void naive_matmul(int *A, int *B, int *C, int N){
    for( int i = 0 ; i < N ; i++ ){
      for( int j = 0 ; j < N ; j++ ){
        for( int k = 0 ; k < N ; k++ ){
          C[i*N + j] += A[i*N + k] * B[k*N + j];
        }
      }
    }

}// end naive_matmul

// Matrix multiplication with inverted loop for better cache access
void smart_matmul(int *A, int *B, int *C, int N){

    for(int i = 0 ; i < N ; i++){
      for(int k = 0 ; k < N ; k++){
        for(int j = 0 ; j < N ; j++){
          C[i*N + j] += A[i*N + k] * B[k*N + j];
        }
      }
    }

}// end smart_matmul

int main() {

    // Defining a square matrix of size N X N
    clock_t start, end;
    double cpu_time_used;
    int N = 1024;
    size_t matrix_size = N * N * sizeof(float);

    // allocating matrices.
    int *A = (int*) malloc(matrix_size);
    int *B = (int*) malloc(matrix_size);
    int *C = (int*) malloc(matrix_size);

    // Initialize matrices A and B in the CPU
    for (int i = 0; i < N*N; i++){
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Benchmarking the first implementation
    start = clock();
    naive_matmul(A, B, C, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("naive_matmul took %f seconds to execute \n", cpu_time_used);

    start = clock();
    smart_matmul(A, B, C, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("smart_matmul took %f seconds to execute \n", cpu_time_used);

    free(A);
    free(B);
    free(C);

    return 0;
}
