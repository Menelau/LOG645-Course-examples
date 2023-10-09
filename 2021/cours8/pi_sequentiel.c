/*
 ============================================================================
 Name        : Estimation_PI.c
 Author      : Carlos Vazquez
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
/* File:     pth_pi.c
 * Purpose:  Try to estimate pi using the formula
 *
 *              pi = 4*[1 - 1/3 + 1/5 - 1/7 + 1/9 - . . . ]
 *
 *           This version has a *very serious bug*
 *
 * Compile:  gcc -g -Wall -o pth_pi pth_pi.c -lm -lpthread
 * Run:      ./pth_pi <number of threads> <n>
 *           n is the number of terms of the series to use.
 *           n should be evenly divisible by the number of threads
 * Input:    none
 * Output:   Estimate of pi as computed by multiple threads, estimate
 *           as computed by one thread, and 4*arctan(1).
 *
 * Notes:
 *    1.  The radius of convergence for the series is only 1.  So the
 *        series converges quite slowly.
 *
 * IPP:   Section 4.4 (pp. 162 and ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>


long long n;
double sum;


/* Only executed by main thread */
void Usage(char* prog_name);
double Serial_pi(long long n);

int main(int argc, char* argv[]) {

    printf("Number of terms to compute: ");
    scanf("%lld", &n);

    printf("With n = %lld terms,\n", n);
    sum = Serial_pi(n);
    printf("   Single thread pi  = %.15f\n", sum);
    printf("                pi = %.15f\n", 4.0*atan(1.0));

    return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:   Serial_pi
 * Purpose:    Estimate pi using 1 thread
 * In arg:     n
 * Return val: Estimate of pi using n terms of Maclaurin series
 */
double Serial_pi(long long n) {
    double sum = 0.0;
    long long i;
    double factor = 1.0;

    for (i = 0; i < n; i++, factor = -factor) {
        sum += factor/(2*i+1);
    }
    return 4.0*sum;

}  /* Serial_pi */
