#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex1;
pthread_mutex_t mutex2;

void* thread_func1(void* args) {
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: locked mutex1\n");
    sleep(2);  // Simulate some work

    printf("Thread 1: waiting for mutex2\n");
    pthread_mutex_lock(&mutex2);  // This causes deadlock
    printf("Thread 1: locked mutex2\n");

    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);

    return NULL;
}

void* thread_func2(void* args) {
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: locked mutex2\n");
    sleep(2);  // Simulate some work

    printf("Thread 2: waiting for mutex1\n");

    pthread_mutex_lock(&mutex1);
    printf("Thread 2: locked mutex1\n");

    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);

    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    pthread_mutex_init(&mutex1, NULL);
    pthread_mutex_init(&mutex2, NULL);

    pthread_create(&thread1, NULL, thread_func1, NULL);
    pthread_create(&thread2, NULL, thread_func2, NULL);

    //wait for threads to finish and continue the program.
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    printf("Joined threads 1 and 2,\n");

    //free the mutexes.
    pthread_mutex_destroy(&mutex1);
    pthread_mutex_destroy(&mutex2);

    return 0;
}
