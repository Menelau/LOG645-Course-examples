#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Matrix size (Squared)
#define N 4

void multiply(int rank, int size, int matrixA[][N], int matrixB[][N], int result[][N]) {
    int i, j, k;
    int start = rank * (N / size);
    int end = (rank + 1) * (N / size);

    #pragma omp parallel for private(j, k) shared(matrixA, matrixB, result)
    for (i = start; i < end; i++) {
        for (j = 0; j < N; j++) {
            result[i][j] = 0;
            for (k = 0; k < N; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size, provided;
    int matrixA[N][N], matrixB[N][N], result[N][N];

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            printf("Error: MPI does not support the required thread level.\n");
        }
        MPI_Finalize();
        return -1;
    }

    // Initialize matrices
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrixA[i][j] = i + 1;
                matrixB[i][j] = j + 1;
            }
        }
    }

    // Broadcast matrices to all processes
    MPI_Bcast(matrixA, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    multiply(rank, size, matrixA, matrixB, result);

    // Gather results at root process
    MPI_Gather(result[rank * (N / size)], N * (N / size), MPI_INT,
               result, N * (N / size), MPI_INT, 0, MPI_COMM_WORLD);

    // Print result at root process
    if (rank == 0) {
        printf("Resultant Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", result[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
