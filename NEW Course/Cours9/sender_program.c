#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    printf("Starting sender program.\n");
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("My rank sender %d\n", world_rank);

    if (world_rank == 0) {
        const int message = 123;
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent message %d\n", message);
    }

    MPI_Finalize();
    return 0;
}
