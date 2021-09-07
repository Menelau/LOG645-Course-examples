/*
 *
 * MPI Scatter and Gather examples.
 *
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){

    int rank;
    int size;

    int sendcount = 1;
    int recvcount = sendcount;
    int sendbuf[4];
    int recvbuf;
    int finalbuf[4];
    int ROOT = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    if (rank == 0){
        sendbuf[0] = 3;
        sendbuf[1] = 5;
        sendbuf[2] = 7;
        sendbuf[3] = 9;
    }

    // Send array to all processes
    MPI_Scatter (sendbuf, sendcount, MPI_INT, &recvbuf, recvcount, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Multiply the received value to its rank
    int local_value = recvbuf * rank;
    printf("Rank: %d, value: %d\n", rank, local_value);

    // Gathering values to ROOT process
    MPI_Gather(&local_value, 1, MPI_INT, finalbuf, 1, MPI_INT, ROOT,
               MPI_COMM_WORLD);

    if (rank==0){
        printf ("Rank: %d, Final_buf array: %d %d %d %d\n", rank,
            finalbuf[0], finalbuf[1], finalbuf[2], finalbuf[3]);
    }

    // int final_sum;
    // MPI_Allreduce(&local_value, &final_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // printf("Rank: %d, Final sum: %d\n", rank, final_sum);
    MPI_Finalize();
    return 0;
}
