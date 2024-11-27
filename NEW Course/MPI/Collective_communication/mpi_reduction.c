#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT 0

int main(){

    int rank, size, buf, res;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    switch(rank){
    case 0: buf = 8; break;
    case 1: buf = 3; break;
    case 2: buf = 7; break;
    case 3: buf = 1; break;
    default: buf = 12; break;
    }    

    MPI_Reduce(&buf, &res, 1, MPI_INT, MPI_MIN, ROOT, MPI_COMM_WORLD);

    if(ROOT == rank){
   printf("%d\n", res);
    }    

    MPI_Finalize();
    return EXIT_SUCCESS;
}

