#include "stdio.h"
#include <immintrin.h>

 int main(void){

    double tab1[4] = {1.76, 9.2, 1.22, 1.22};
       double tab2[4];
       __m256d a, c;

       a = _mm256_loadu_pd( tab1 );
       c = _mm256_set_pd( 2.0, 3.0, 2.0, 3.0 );
       a = _mm256_mul_pd( a, c );
       _mm256_storeu_pd( tab2, a );

       printf( "%.2lf %.2lf %.2lf %.2lf\n", tab2[0], tab2[1], tab2[2], tab2[3] );

    }
