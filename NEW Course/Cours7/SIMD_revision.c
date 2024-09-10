#include <stdio.h>
#include <immintrin.h> // for AVX intrinsics

void process_array(float *array, int size) {
    __m256 one_vector = _mm256_set1_ps(1.0);
    __m256 zero_vector = _mm256_set1_ps(0.0);

    for (int i = 0; i < size; i += 8) {
       __m256 vec = _mm256_loadu_ps(&array[i]);
       // Create mask of negative numbers.
       __m256 neg_mask = _mm256_cmp_ps( vec, zero_vector, _CMP_LE_OQ);
       //blend results.
       __m256 result = _mm256_blendv_ps(zero_vector, one_vector, neg_mask);
       _mm256_storeu_ps(&array[i], result);
    }

}

int main() {
    float numbers[] = {-1.2, 2.3, -3.4, 4.5, -5.6, 6.7, -7.8, 8.9};
    int size = sizeof(numbers) / sizeof(numbers[0]);
    process_array(numbers, size);
    for (int i = 0; i < size; i++) {
        printf("%d ", (int) numbers[i]);
    }
    
    return 0;
}
