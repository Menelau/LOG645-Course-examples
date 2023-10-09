#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv){

    printf("Initializing\n");
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank > 0){
        printf("Employe effectue un travail intensif.\n");
    } else if (rank == 0){
        printf("Patron affiche les resultats.\n");
    }

    MPI_Finalize();
    printf("Finalizing\n");
    return 0;
}
