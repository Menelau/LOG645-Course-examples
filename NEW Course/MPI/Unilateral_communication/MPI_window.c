#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    int rank, peer;
    int* token;
    MPI_Win win;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate memory for the window
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &token, &win);
    *token = getpid();

    // Synchronize before the RMA operation
    MPI_Win_fence(0, win);

    if (rank == 0) {
        peer = 1;
        int received_token;
        // Fetch the token value from rank 1
        MPI_Get(&received_token, 1, MPI_INT, peer, 0, 1, MPI_INT, win);
        printf("Rank %d fetched token value %d from rank %d\n", rank, received_token, peer);
    }

    // Synchronize after the RMA operation
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);
    MPI_Finalize();

    return 0;
}
