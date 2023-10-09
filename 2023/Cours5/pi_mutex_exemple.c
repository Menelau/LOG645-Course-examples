/*
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
#include "timer.h"

// global variables. They are shared among all threads
long thread_count;
long long n;
double sum;
int flag;

// creating the mutex as a global variable to share between threads.
pthread_mutex_t mutex;

void* Thread_Sum_Mutex(void* rank);

int main(int argc, char* argv[]) {
    long       thread;  /* Use long in case of a 64-bit system */
    pthread_t* thread_handles;

    //get number of threads from input args
    thread_count = strtol(argv[1], NULL, 10);

    //get number of iterations from input args
    n = strtol(argv[2], NULL, 10);

    printf("thread count %ld\n", thread_count);

    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t));
    sum = 0.0;

    // Create the mutex before runnig the threads (important as the mutex will be used inside the thread function)
    pthread_mutex_init(&mutex, NULL);

    // Create all threads
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL,
                Thread_Sum_Mutex, (void*)thread);

    // Wait for all threads to finish before continuing
    for (int thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

    sum = 4.0 * sum;

    printf("With n = %lld terms and thread_count = %ld,\n", n, thread_count);
    printf("   Our estimate of pi = %.15f\n", sum);
    printf("                   pi = %.15f\n", 4.0*atan(1.0));

    // Destroy the mutex to free up the resources.
    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    return 0;
}


void* Thread_Sum_Mutex(void* rank) {
    double my_sum = 0;
    long my_rank = (long) rank;
    double factor;
    long long i;
    long long my_n = n / thread_count;
    long long my_first_i = my_n * my_rank;
    long long my_last_i = my_first_i + my_n;

    if (my_first_i % 2 == 0){
        factor = 1.0;
    } else {
        factor = -1.0;
    }

    for (i = my_first_i; i < my_last_i; i++, factor = -factor){
        my_sum += factor / (2 * i + 1);
    }

    // Locking the sum with a mutex.
    pthread_mutex_lock(&mutex);
    sum += my_sum;
    pthread_mutex_unlock(&mutex);

    return NULL;
}
