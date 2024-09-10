#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t lock;
pthread_cond_t condition;
int buffer = 0;

void* Add_Frames_Buffer(void* arg) {
    int iterations = 10;

    for (int i = 0; i < iterations; i++) {
        //protects writing operation on the shared resources with a mutex
        pthread_mutex_lock(&lock);
        buffer += 10;
        printf("Adding to buffer... %d\n", buffer);
        pthread_mutex_unlock(&lock);

        //signal the other threads.
        pthread_cond_signal(&condition);
        // sleep to give its "quantum" to the other threads.
        sleep(1);
    }
    return NULL;
}

void* Process_Frames(void* arg) {
    pthread_mutex_lock(&lock);
    while (buffer < 50) {
        printf("Waiting for 50 frames in the buffer...\n");
        // wait here for a cond_signal on the same variable "condition"
        pthread_cond_wait(&condition, &lock);

    }
    sleep(1);    //intensive calculation
    buffer = buffer - 50;
    printf("Left %d frames in the buffer\n", buffer);
    pthread_mutex_unlock(&lock);
    return NULL;
}

int main(int argc, char* argv[]) {

    //allocating trheads
    pthread_t thread_process_frames;
    pthread_t thread_collect_frames;

    //creating mutex and conditional variable
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&condition, NULL);

    pthread_create(&thread_process_frames, NULL, &Process_Frames, NULL);
    pthread_create(&thread_collect_frames, NULL, &Add_Frames_Buffer, NULL);

    // waiting for threads to finish
    pthread_join(thread_collect_frames, NULL);
    pthread_join(thread_process_frames, NULL);

    //clear the mutex and conditional variable
    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&condition);
    return 0;
}
