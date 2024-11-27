#include <mpi.h>
#include <stdio.h>
#include <unistd.h> // for getpid()

int main(int argc, char* argv[]) {
    int rank, peer;
    int* token;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate memory for the window
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &token, &win);
    *token = getpid();

    if (rank == 0) {
        peer = 1;
        int received_token;

        // Lock the window of the target rank
        MPI_Win_lock(MPI_LOCK_SHARED, peer, 0, win);

        // Fetch the value of the token from the target rank's window
        MPI_Get(&received_token, 1, MPI_INT, peer, 0, 1, MPI_INT, win);

        // Unlock the target rank's window
       MPI_Win_unlock(peer, win);

        // Print the fetched value
        printf("Rank %d fetched token value %d from rank %d\n", rank, received_token, peer);
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
