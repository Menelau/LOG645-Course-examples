/* Exercice 1 - parallelize code */

#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "math.h"


int main(void){

    int row = 10000;
    int col = 10000;
    static float a[10000][10000];
    time_t start, end;
    srand(time(0));          // set random seed

    time(&start);

    // Initializing matrix with random values
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            float num = ((float) rand()/(float)(RAND_MAX)) * 1.0;
            a[i][j] = num;
        }
    }

    // Calculation to be parallelized.
    for(int i = 1; i < row; i++){
        for(int j = 0; j < col; j++){
            float tmp = 2.0 * a[i-1][j];
            tmp = pow(tmp, 2.0);
            tmp = tmp/3.0;
            a[i][j] = tmp;
        }
    }

    time(&end);
    double time_taken = difftime(end, start);
    printf("Elapsed time: %f", time_taken);

}
