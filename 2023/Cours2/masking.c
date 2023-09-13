/* Masking example

 In performance-critical applications, masking can eliminate branches and enable
 efficient data transformations when combined with SIMD operations, making it an
 important tool for optimizing code execution.

Note: as mentioned during classes, masking is an extremely important concept
to master. It is used even if you do programming in a more high level like
Python with vectorized code (NumPy).

*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//AVX include. You can try changing it to SIMDe if you have a M1 or M2 machine.
/include <immintrin.h>

int main(){

    double in[8] = {-3.1415, 29.823, -0.452, 48.8, 37.77777, -13,328, -996.0};
    double out[8] = {0,0,0,0,0,0,0,0};
    double absolute[8] = {0,0,0,0,0,0,0,0};

    __m256d zero = _mm256_setzero_pd();
    __m256d negative = _mm256_set1_pd( -1.0 );
    __m256d data, mask, blend;

    // Negative elements are zeroed.
    for(int i = 0 ; i < 8 ; i += 4) {
       data = _mm256_loadu_pd( in+i );
       mask = _mm256_cmp_pd( data, zero, _CMP_LE_OQ);
       blend = _mm256_blendv_pd( data, zero, mask );
       _mm256_storeu_pd( out+i, blend );
    }

    printf( "Zeroing negative elements: " );
    for(int i = 0 ; i < 8 ; i++){
        printf( "%.4lf  ", out[i] );
    }
    printf( "\n" );

    // Absolute value is returned.
    __m256d moinsun = _mm256_set1_pd( -1.0 );
    for( int i = 0 ; i < 8 ; i+=4 ) {
       data = _mm256_loadu_pd( in+i );
       mask = _mm256_cmp_pd( data, zero, _CMP_LE_OQ);
       __m256d inverted = _mm256_mul_pd( data, negative );
       blend = _mm256_blendv_pd( data, inverted, mask );
       _mm256_storeu_pd( out+i, blend );
    }

    printf( "Absolute value: " );
    for(int i = 0 ; i < 8 ; i++){
        printf( "%.4lf  ", out[i] );
    }
    printf( "\n" );
    return 0;
}
