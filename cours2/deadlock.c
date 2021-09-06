/* simple deadlock */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv){
    int my_rank;
    MPI_Status status;
    double a[1000], b[1000];

    MPI_init(&argc, &argv);
    MPI_COMM_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0){
        MPI_Recv(b, 1000, MPI_DOUBLE, 1, 19, MPI_COMM_WORLD, &status);
        MPI_Send(a, 1000, MPI_DOUBLE, 1, 17, MPI_COMM_WORLD);
    } else if (my_rank == 1){
        MPI_Recv(b, 1000, MPI_DOUBLE, 0, 17, MPI_COMM_WORLD, &status);
        MPI_Send(a, 1000, MPI_DOUBLE, 0, 19, MPI_COMM_WORLD);
    }

    //terminate
    MPI_Finalize();
}
