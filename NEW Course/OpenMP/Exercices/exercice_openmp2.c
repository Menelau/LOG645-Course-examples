#include "stdio.h"

int main(void){
    int total = 0;
    #pragma omp parallel
    for (int i = 0; i < 100; i++){
    }

    printf("total : %d", total);

    return 0;
}
