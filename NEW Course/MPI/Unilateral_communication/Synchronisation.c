#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, i, tmp;
    int* shared_array;
    int* ranks; // Array for ranks
    MPI_Win win;
    MPI_Group group, world_group;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("test100");

    // Allocate memory for shared array
    shared_array = (int*) malloc(sizeof(int) * size);
    for (i = 0; i < size; i++) {
        shared_array[i] = 0;
    }

    // Create an MPI window for the shared array
    MPI_Win_create(shared_array, size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Create an array of ranks to include in the group
    ranks = (int*) malloc(sizeof(int) * size);
    for (i = 0; i < size; i++) {
        ranks[i] = i;
    }
    // Get the world group and create a new group including all ranks
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, size, ranks, &group);

    // Synchronize to ensure all targets have posted before the origin starts
    MPI_Barrier(MPI_COMM_WORLD);

        // All processes (targets) call MPI_Win_post before the origin calls MPI_Win_start
    MPI_Win_post(group, 0, win);

    if (rank == 0) {
        // Origin process: initiate an access epoch after targets have posted
        MPI_Win_start(group, 0, win);

        // Write the rank value to all other processes
        tmp = rank;
        for (i = 0; i < size; i++) {
            MPI_Put(&tmp, 1, MPI_INT, i, rank, 1, MPI_INT, win);
        }

        // Complete the access epoch
        MPI_Win_complete(win);
    }

    // All processes wait for the RMA operations to complete
    MPI_Win_wait(win);

    // Verify the result
    printf("Rank %d: shared_array[%d] = %d\n", rank, rank, shared_array[rank]);

    // Free resources
    free(ranks);
    MPI_Group_free(&group);
    MPI_Group_free(&world_group);
    MPI_Win_free(&win);
    free(shared_array);

    MPI_Finalize();
    return 0;
}
