#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int provided;

    // Initialize MPI with a requested thread safety level
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Print the thread safety level provided
    switch (provided) {
        case MPI_THREAD_SINGLE:
            printf("MPI_THREAD_SINGLE: Only one thread will execute.\n");
            break;
        case MPI_THREAD_FUNNELED:
            printf("MPI_THREAD_FUNNELED: Only the main thread will make MPI calls.\n");
            break;
        case MPI_THREAD_SERIALIZED:
            printf("MPI_THREAD_SERIALIZED: Multiple threads may make MPI calls, but only one at a time.\n");
            break;
        case MPI_THREAD_MULTIPLE:
            printf("MPI_THREAD_MULTIPLE: Multiple threads may make MPI calls without restrictions.\n");
            break;
        default:
            printf("Unknown thread safety level.\n");
    }

    MPI_Finalize();
    return 0;
}
