#include <immintrin.h> //header for AVX
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 64

double horizontal_add(__m256d vec) {
    // Shuffle and add horizontally (combining elements within the register)
    __m256d temp = _mm256_hadd_pd(vec, vec);
    __m128d high = _mm256_extractf128_pd(temp, 1);
    __m128d low = _mm256_castpd256_pd128(temp);
    __m128d sum = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(sum);
}

int main() {
    // Allocate memory for the vectors, ensuring 32-byte alignment (for AVX)
    double* a = (double*)aligned_alloc(32, VECTOR_SIZE * sizeof(double));
    double* b = (double*)aligned_alloc(32, VECTOR_SIZE * sizeof(double));
    double* result = (double*)aligned_alloc(32, VECTOR_SIZE * sizeof(double));

    // Initialize vectors a and b (from 0 to 63, from 63 to 0)
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (double) i;
        b[i] = (VECTOR_SIZE - i) * 1.0;
    }

    // Vertical addition: Perform element-wise addition in chunks of 4 doubles (256 bits)
    printf("Vertical addition results (element-wise):\n");
    for (int i = 0; i < VECTOR_SIZE; i += 4) {
        // Load 256 bits (4 doubles) from each input array into AVX registers
        __m256d vec_a = _mm256_load_pd(&a[i]);
        __m256d vec_b = _mm256_load_pd(&b[i]);

        __m256d vec_result = _mm256_add_pd(vec_a, vec_b);
        _mm256_store_pd(&result[i], vec_result);
        printf("Result[%d-%d] = [%f, %f, %f, %f]\n", i, i+3,
               result[i], result[i+1], result[i+2], result[i+3]);
    }

    // Horizontal operation: Sum elements in each 256-bit chunk
    printf("\nHorizontal addition of each 256-bit result (summing 4 elements):\n");
    for (int i = 0; i < VECTOR_SIZE; i += 4) {
        __m256d vec_result = _mm256_load_pd(&result[i]);

        double sum = horizontal_add(vec_result);
        printf("Sum of result[%d-%d] = %f\n", i, i+3, sum);
    }

    free(a);
    free(b);
    free(result);

    return 0;
}
