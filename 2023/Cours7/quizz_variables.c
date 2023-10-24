#include "stdio.h"

int main(void){

    int x = 5;
    #pragma omp parallel for lastprivate(x)
    for (int i = 0; i < 10; i++) {
        x += i;
    }
    printf("La valeur de x est : %d", x);
}
