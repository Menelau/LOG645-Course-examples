#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int N = 1000;
    double A[N][N];
    double traceA = 0;
    srand(time(0));             //setting random seed

    // initializing matrix A with random values.
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            double num = ((double) rand()/(double) RAND_MAX) * 1.0;
            A[i][j] = num;
        }
    }
    #pragma omp parallel for default(shared) reduction(+:traceA)
    for (int i = 0; i < N; i++) {
        traceA += A[i][i];
    }

    printf("The trace of Matrix A is: %f\n", traceA);
    return 0;

 }

