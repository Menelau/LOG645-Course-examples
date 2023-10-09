#include <stdio.h>
#include <stdlib.h>
#include "stdbool.h"

int main(int argc, char **argv){
    int n_iter = 1000;
    bool is_prime[n_iter];

    #pragma omp parallel for schedule(dynamic)
    for(int index=0; index < n_iter; index++){
        long potential_prime = rand() % (4000000000 + 1);

        for (long multiple = 2; multiple < potential_prime; multiple++){
            if ((potential_prime % multiple) == 0){
                is_prime[index] = false;
                break;
            }
        }

    }
    return 0;
}
