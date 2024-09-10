#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/wait.h>

#define ARRAY_SIZE 16
#define SEGMENTS 4

void compute(double *arr, int start, int end) {
    for (int i = start; i < end; i++) {
        arr[i] = arr[i]*arr[i];
    }
}

int main() {
    double array[ARRAY_SIZE];
    // Initialize the array with some values.
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = i;
    }

    int segment_size = ARRAY_SIZE / SEGMENTS;

    for (int s = 0; s < SEGMENTS; s++) {
        pid_t pid = fork();

        if (pid == 0) {  // Child process
            compute(array, s * segment_size, (s + 1) * segment_size);
            printf("Segment %d processed by child with PID %d with parent %d\n", s, getpid(), getppid());
        }
    }
    // Parent process waits for all child processes to complete
    for (int s = 0; s < SEGMENTS; s++) {
        wait(NULL);
    }
    printf("\n");
    return 0;
}
