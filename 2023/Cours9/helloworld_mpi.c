#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank;
    int world;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    printf("Hello: rank %d, world: %d\n",rank, world);
    MPI_Finalize();
    return 0;

}
