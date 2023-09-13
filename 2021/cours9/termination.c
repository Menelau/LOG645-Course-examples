#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <semaphore.h>

const long thread_count = 2;

void* Thread_Func1(void* args);
void* Thread_Func2(void* args);

int main(int argc, char* argv[]) {

    pthread_t thread1;
    pthread_t thread2;

    long* return_threads[2];

    // Running threads 1 and 2
    pthread_create(&thread1, NULL, &Thread_Func1, (void*) 1);
    pthread_create(&thread2, NULL, &Thread_Func2, (void*) 2);

    // Joining threads and collecting their results.
    pthread_join(thread1, (void**) &return_threads[0]);
    pthread_join(thread2, (void**) &return_threads[1]);

    printf("Value computed by thread 1: %ld\n", *return_threads[0]);
    printf("Value computed by thread 2: %ld\n", *return_threads[1]);

    //freeing resources allocated dynamically (in the heap).
    free(return_threads[0]);
    free(return_threads[1]);
    return 0;
}

void* Thread_Func1(void* args) {
    long rank = (long) args;
    long* value = malloc(sizeof(long));
    sleep(5);
    *value = 10 * rank;
    printf("Thread %ld ending execution.\n", rank);

    //Terminating execution using the standard C return command
    return (void*) value;
}

void* Thread_Func2(void* args) {
    long rank = (long) args;
    long* value = malloc(sizeof(long));
    *value = 10 * rank;
    printf("Thread %ld ending execution.\n", rank);

    //Terminating execution using the thread_exit function from pthreads
    pthread_exit((void*) value);
}
