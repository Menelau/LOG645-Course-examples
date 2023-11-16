/* simple deadlock */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]){
    int my_rank;
    MPI_Status status;
    double a[1000], b[1000];

    for(int i=0; i< 1000; i++){
        a[i] = 0;
        b[i] = 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0){
        MPI_Send(a, 1000, MPI_DOUBLE, 1, 17, MPI_COMM_WORLD);
        printf("Process %d sent its message.\n", my_rank);
        MPI_Recv(b, 1000, MPI_DOUBLE, 1, 17, MPI_COMM_WORLD, &status);
        printf("Process %d received its message.\n", my_rank);

    } else if (my_rank == 1){
        MPI_Recv(b, 1000, MPI_DOUBLE, 0, 19, MPI_COMM_WORLD, &status);
        printf("Process %d sent its message.\n", my_rank);
        MPI_Send(a, 1000, MPI_DOUBLE, 0, 19, MPI_COMM_WORLD);
        printf("Process %d received its message.\n", my_rank);

    }
    if(my_rank == 0){
        printf("End process.\n");
    }
    printf("process %d finished\n", my_rank);
    MPI_Finalize();
    return 0;
}
