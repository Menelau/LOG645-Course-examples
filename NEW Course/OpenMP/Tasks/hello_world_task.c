#include <stdio.h>
#include <stdlib.h>

int main(void) {

#pragma omp parallel
    {
        #pragma omp task
        printf("Hello world!\n");
    }
    return EXIT_SUCCESS;
}
