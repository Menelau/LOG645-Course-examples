/*
Sequential code to compute the integral f(x) = 4.0/1 + x^2. From 0 to 1.

Solving the integral numerically gives the value of Pi.

*/

#include <stdio.h>
#include "time.h"


int main(void){

    // long num_steps = 1000000000;
    // long num_steps = 100000000;
    double step;
    double x, integral;
    double sum = 0;

    time_t start, end;
    time(&start);
    step = 1.0/ (double) num_steps;

    #pragma omp parallel for
    for (int i = 0; i < num_steps; i++){
        x = (i + 0.5) * step;

        #pragma omp critical
        sum = sum + 4.0/(1.0 + (x*x));
    }

    integral = step * sum;
    time(&end);
    double time_taken = difftime(end, start);
    printf("Integral: %f\n", integral);
    printf("Elapsed time: %f\n", time_taken);

}

