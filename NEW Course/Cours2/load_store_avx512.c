#include <stdio.h>
#include <immintrin.h>

int main() {
    int data[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    };

    //selecting which indices, from the array we want to obtain.
    __m256i indices = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);

    // Here, scale is set to 4 because our base type is int which is 4 bytes.
    __m256i result = _mm256_i32gather_epi32(data, indices, 4);

    //copying it back to an array. We could do the same later.
    int output[8];
    _mm256_storeu_si256((__m256i*)output, result);

    for (int i = 0; i < 8; i++) {
        printf("%d ", output[i]);
    }

    int output2[32] = {0};
    _mm256_i32scatter_epi32(output2, indices, result, 4);

    for (int i = 0; i < 32; i++) {
        printf("%d ", output2[i]);
    }
    return 0;
}

