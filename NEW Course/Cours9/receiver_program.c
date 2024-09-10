#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    printf("Starting receiver program.\n");
    MPI_Init(&argc, &argv);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf("My rank receiver %d\n", my_rank);

    if (my_rank == 1) {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received message %d\n", message);
    }

    MPI_Finalize();
    return 0;
}
