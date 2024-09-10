#include <stdio.h>


int main(void){
    
    int total = 0;
    int n_iter = 1000000;

    #pragma omp parallel for
    for(int i=0; i<n_iter; i++){
        total++;
    }
    printf("Value: %d", total);

    return 0;
}
