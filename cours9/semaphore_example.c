/*
 * Semaphore example
 *
 * This code shows an usage of semaphore in a producer-consummer application.
 *
 * Run: ./semaphore
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>         // include for the sleep function


// As usual our beautiful global variables to share between threads
int buffer[10];
int resource_count = 0;
long thread_count;

// variables for mutex and semaphores
sem_t empty;
sem_t full;
pthread_mutex_t lock_prod_cons;

// Thread functions for producer and consumer
void* Producer_Func(void* args);
void* Consumer_Func(void* args);


int main(int argc, char* argv[]) {

    thread_count = strtol(argv[1], NULL, 10);

    pthread_t *thread_handles = NULL;
    thread_handles = malloc(sizeof(thread_handles)*thread_count);

    // initialize semaphores and threads
    sem_init(&empty, 0, 10);
    sem_init(&full, 0, 0);
    pthread_mutex_init(&lock_prod_cons, NULL);


    for (int thread = 0; thread < thread_count; thread++) {
        // If i == 0 creates a consumer thread. otherwise create producer threads
        if (thread > 0) {
            pthread_create(&thread_handles[thread], NULL, &Producer_Func, NULL);
        } else {
            pthread_create(&thread_handles[thread], NULL, &Consumer_Func, NULL);
        }
    } //END for

    // Wait for threads to finish
    for (int thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    // destroy all synchronization mechanisms
    sem_destroy(&empty);
    sem_destroy(&full);
    pthread_mutex_destroy(&lock_prod_cons);
    free(thread_handles);
    return 0;
}


void* Producer_Func(void* args) {
    while (true) {
        // Produce
        int x = rand() % 100;
        sleep(1);

        // Add to the buffer
        sem_wait(&empty);
        pthread_mutex_lock(&lock_prod_cons);
        buffer[resource_count] = x;
        resource_count++;
        pthread_mutex_unlock(&lock_prod_cons);
        sem_post(&full);
    }
}

void* Consumer_Func(void* args) {
    while (true) {
        int y;

        sem_wait(&full);
        pthread_mutex_lock(&lock_prod_cons);
        y = buffer[resource_count - 1];
        resource_count--;

        pthread_mutex_unlock(&lock_prod_cons);
        sem_post(&empty);
        printf("Got %d\n", y);
        sleep(1);
    }
}
