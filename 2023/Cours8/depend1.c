#include <stdio.h>
#include <stdlib.h>

int main() {
    int x = 1;
#pragma omp parallel
#pragma omp single
    {
#pragma omp task shared(x) depend(out: x)
        x = 2;
#pragma omp task shared(x) depend(in: x)
        printf("x = %d\n", x);
    }
    return EXIT_SUCCESS;
}

