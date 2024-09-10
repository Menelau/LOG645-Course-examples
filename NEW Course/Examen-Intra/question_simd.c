#include <stdio.h>
#include <immintrin.h> // for AVX intrinsics

const double C = 0.5; // Change this value according to your needs

void process_array_simd(double *array, long size) {
    __m128d C_vector = _mm_set1_pd(C);
    __m128d zero_vector = _mm_set1_pd(0.0);
    __m128d sign_mask = _mm_set1_pd(-0.0); 

    long i;
    for (i = 0; i <= size - 4; i += 4) {
        __m256d vec = _mm256_loadu_pd(&array[i]);

        // Create mask of negative numbers
        __m256d neg_mask = _mm256_and_pd(vec, _mm256_castpd128_pd256(sign_mask));
        neg_mask = _mm256_cmp_pd(neg_mask, _mm256_castpd128_pd256(sign_mask), _CMP_EQ_OQ);
        
        // Multiply numbers with C
        __m256d multiplied = _mm256_mul_pd(vec, _mm256_castpd128_pd256(C_vector));
        
        // Blend values: use zero_vector for negative numbers, and multiplied for non-negative
        __m256d result = _mm256_blendv_pd(multiplied, _mm256_castpd128_pd256(zero_vector), neg_mask);
        
        _mm256_storeu_pd(&array[i], result);
    }
    
    // Handle any remaining elements
    for (; i < size; i++) {
        if (array[i] < 0) {
            array[i] = 0;
        } else {
            array[i] *= C;
        }
    }
}

int main() {
    double numbers[] = {-1.2, 2.3, -3.4, 4.5, -5.6, 6.7, -7.8, 8.9};
    long size = sizeof(numbers) / sizeof(numbers[0]);
    
    process_array_simd(numbers, size);
    
    for (long i = 0; i < size; i++) {
        printf("%f ", numbers[i]);
    }
    
    return 0;
}
