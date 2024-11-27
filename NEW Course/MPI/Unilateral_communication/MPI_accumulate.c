#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){
    int rank, size, *a, *b, i;
    MPI_Win win;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory for arrays
    MPI_Alloc_mem(sizeof(int) * size, MPI_INFO_NULL, &a);
    MPI_Alloc_mem(sizeof(int) * size, MPI_INFO_NULL, &b);

    for (i = 0; i < size; i++) {
        // Initialize a with value based on rank and index
        a[i] = rank * 100 + i;
        // Initialize b, the shared array, to 0
        b[i] = 0;
    }

    // Create an MPI window for the shared array b"
    MPI_Win_create(b, size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Synchronize before accumulation starts
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win);

    // Accumulate each process's contributions into the target processes
    for (i = 0; i < size; i++) {
        MPI_Accumulate(&a[i], 1, MPI_INT, i, rank, 1, MPI_INT, MPI_SUM, win);
    }

    // MPI_Fence to synchronize after accumulation is complete
    MPI_Win_fence((MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), win);

    printf("Rank %d: Accumulated values in b: ", rank);
    for (i = 0; i < size; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");

    MPI_Win_free(&win);
    MPI_Free_mem(a);
    MPI_Free_mem(b);
    MPI_Finalize();
    return 0;
}
