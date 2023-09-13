#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[]){
    int size;
    int rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // Get my rank and do the corresponding job
    if (rank == 0){
        // The "master" MPI process sends the message.
        int buffer = 42;
        printf("[Process %d] I send the value %d.\n", rank, buffer);
        for (int i=1; i<size; i++ ){
            MPI_Ssend(&buffer, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

    } else {
        int received;
        MPI_Request request;
        printf("[Process %d] I issue the MPI_Irecv to receive the message as a background task.\n", rank);
        MPI_Irecv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
 
        // other instructions while Irecv is completed
        printf("[Process %d] The MPI_Irecv is issued, I now moved on to print this message.\n", rank);
 
        // Wait for the MPI_Recv to complete.
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        printf("[Process %d] The MPI_Irecv completed, therefore so does the underlying MPI_Recv. I received the value %d.\n", rank, received);

    }
    MPI_Finalize();
    return 0;
}
