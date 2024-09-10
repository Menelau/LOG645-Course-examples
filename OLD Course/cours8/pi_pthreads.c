/*
 */
/* Code to estimate Pi using the formula:
 *
 *              pi = 4*[1 - 1/3 + 1/5 - 1/7 + 1/9 - . . . ]
 *
 * This version has a very serious race condition bug. It is only used for
 * demonstration purposes.
 *
 * Run:      ./pth_pi <number of threads> <n>
 *
 * Code inspired by Introduction to Parallel Programming chapter 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

// global variables. They are shared among all threads
long thread_count;
long long n;
double sum;

void* Thread_Sum(void* rank);

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

    //create all threads
    for (long thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL,
                Thread_Sum, (void*)thread);

    // Wait for all threads to finish before continuing
    for (int thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

    sum = 4.0 * sum;

    printf("With n = %lld terms and thread_count = %ld,\n", n, thread_count);
    printf("   Our estimate of pi = %.15f\n", sum);
    printf("                   pi = %.15f\n", 4.0*atan(1.0));

    free(thread_handles);
    return 0;
}

/**************
Thread function
**************
*/
void* Thread_Sum(void* rank) {

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
        sum += factor / (2 * i + 1);
    }

    return NULL;
}
