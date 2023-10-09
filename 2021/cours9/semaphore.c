#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <semaphore.h>

long thread_count;
sem_t semaphore;

void* Semaphore_Function(void* args) {
    long rank = (long) args;
    sem_wait(&semaphore);
    printf("Hello from thread %ld.\n", rank);
    sleep(10);   // simulating intensive calculation...
    printf("Thread %ld returning token.\n", rank);
    sem_post(&semaphore);
    return NULL;
}

int main(int argc, char *argv[]) {

    thread_count = strtol(argv[1], NULL, 10);

    pthread_t *thread_handles;
    thread_handles = malloc(thread_count * sizeof(pthread_t));

    // initializing a semaphore with 4 "tokens" or threads that can run in parallel
    sem_init(&semaphore, 0, 4);

    for (long thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, &Semaphore_Function, (void*) thread);
    }

    for (int thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    //destroy semaphore
    sem_destroy(&semaphore);
    free(thread_handles);

    return 0;
}
