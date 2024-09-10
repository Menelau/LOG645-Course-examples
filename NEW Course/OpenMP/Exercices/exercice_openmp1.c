#include "stdio.h"
#include <omp.h>

int main(void){
   // #pragma omp parallel for
   // for (int i = 0; i < 100; ++i){
   //     #pragma omp critical
   //     printf("Iteration number: #%d\n", i);
   // }
   // return 0;
    int k = 10;
    int a[2*10];
    for (int i = k; i < (2 * k); i++) {
        a[i] = a[i] + a[i - k];
    }

}
