#include <stdio.h>
#include <immintrin.h>


int main() {
    int data[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    };
    int output[8];

    __m256i indices = _mm256_setr_epi32(2, 4, 8, 12, 16, 20, 24, 28);
    __m256i result = _mm256_i32gather_epi32(data, indices, 4);

    _mm256_storeu_si256((__m256i*)output, result);
    printf("Output: ");
    for (int i = 0; i < 8; i++) {
        printf("%d \n", output[i]);
    }
    printf("\n");


    int output2[32] = {0};

    //simulating scatter without AVX512
    int scatter_positions[8] = {1, 5, 9, 13, 17, 21, 25, 29};
    int gathered_values[8];
    _mm256_storeu_si256((__m256i*)gathered_values, result);

    for (int i = 0; i < 8; i++) {
        output2[scatter_positions[i]] = gathered_values[i];
    }

    printf("Output2: ");

    for (int i = 0; i < 32; i++) {
        printf("%d ", output2[i]);
    }

    return 0;
}

/*
int main() {
    int data[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    };

    int output[32] = {0};

    __m256i indices = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    __m256i result = _mm256_i32gather_epi32(data, indices, 4);

    int scatter_positions[8] = {1, 5, 9, 13, 17, 21, 25, 29};
    int gathered_values[8];
    _mm256_storeu_si256((__m256i*)gathered_values, result);

    for (int i = 0; i < 8; i++) {
        output[scatter_positions[i]] = gathered_values[i];
    }

    for (int i = 0; i < 32; i++) {
        printf("%d ", output[i]);
    }

    return 0;
}*/
