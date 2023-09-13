/* Dot product example

Different than the majority of the course examples, this one is using float
instead of double. Look that we can fit more elements in the registers and we
use the _ps (packed single) instad of _pd (packed double).

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// FMA is part of AVX, so we need to add AVX intrinsics
#include <immintrin.h>

// note that I'm not treating the last elements just to simplify the example.
double dot_product_fma(float *A, float *B, int N){
    __m256 prod, a, b;
    prod = _mm256_setzero_ps();
    //as we are using float we can have 8 elements in a 256 register AVX.
    for(int i = 0 ; i < N ; i += 8){
        //pd since the data type is double.
        a = _mm256_loadu_ps( &A[i] );
        b = _mm256_loadu_ps( &B[i] );
        //note that this line does the sum to the prod variable. No need to +=
        prod = _mm256_fmadd_ps(a, b, prod);
    }
    double result;
    result = (float) prod[0];
    result += (float) prod[1];
    result += (float) prod[2];
    result += (float) prod[3];
    return result;
}

int main (){

    // Defining a square matrix of size N X N
    clock_t start, end;
    double cpu_time_used;
    int N = 4096;

    // allocating matrices.
    float *A = (float *) malloc(sizeof(float) * N);
    float *B = (float *) malloc(sizeof(float) * N);

    // Initialize matrices A and B in the CPU
    for (int i = 0; i < N; i++){
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }

    start = clock();
    float prod = dot_product_fma(A, B, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Result: %.4lf. \n", prod);
    printf("SIMD execution took %f seconds to execute. \n", cpu_time_used);

    free(A);
    free(B);
    return 0;

}
