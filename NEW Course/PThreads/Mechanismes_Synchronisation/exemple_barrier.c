#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#ifdef __APPLE__
    #include "osx_barrier.h"
#endif // __APPLE__

// Barrier as a global variable.
pthread_barrier_t barrier;
long thread_count;

void* Thread_Func(void* rank);

int main(int argc, char *argv[]) {

    pthread_t* thread_handles;

    //get number of threads from input args
    thread_count = strtol(argv[1], NULL, 10);

    // allocating memory for all threads
    thread_handles = malloc(thread_count*sizeof(pthread_t));

    // initializing the barrier. Using thread_count to wait for all threads
    // to arrive at the barrier before continuing.
    pthread_barrier_init(&barrier, NULL, 10);

    // creating threads
    for (long i = 0; i < thread_count; i++)
        pthread_create(&thread_handles[i], NULL, &Thread_Func, (void*) i);

    // waiting for all threads to finish.
    for (long i = 0; i < 10; i++)
        pthread_join(thread_handles[i], NULL);

    // calling destroy to free up the barrier resources.
    pthread_barrier_destroy(&barrier);
    return 0;
}

void* Thread_Func(void* arg) {
    long rank = (long) arg;
    printf("Thread: %ld of %ld, before the barrier...\n", rank, thread_count);
    // barrier part inside the thread function code.
    pthread_barrier_wait(&barrier);
    printf("Thread: %ld of %ld, passed the barrier\n",rank , thread_count);

    return NULL;
}
