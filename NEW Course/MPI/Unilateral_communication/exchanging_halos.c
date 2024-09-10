#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define GRID_WIDTH 2
#define GRID_HEIGHT 2

void performComputation(double *local_matrix, int width, int height);
void printMatrix(double *matrix, int width, int height);
void exchangeHalos(double *local_matrix, int width, int height,
                   int up, int down, int left, int right, MPI_Comm cart_comm);

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sendcount = GRID_WIDTH * GRID_HEIGHT;
    int recvcount = sendcount;

    double *matrix;
    double *local_matrix = (double *) malloc(sizeof(double) * recvcount);

    if(rank == 0){
        matrix = (double *) malloc(sizeof(double) * GRID_WIDTH * GRID_HEIGHT);
        double count = 0;
        for(int i = 0; i < GRID_HEIGHT * GRID_WIDTH; i++){
            matrix[i] = count;
            count++;
        }
    }

    MPI_Scatter(matrix, sendcount, MPI_DOUBLE, local_matrix, recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //2D topology, leave MPI decide the configuration
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {1, 1}; // Periodic boundary conditions
    MPI_Comm cart_comm;
    // creating a communicator for our 2D cartesian
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    //int coords[2];
    //MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Neighbors
    enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};
    char* neighbours_names[4] = {"down", "up", "left", "right"};
    int neighbours_ranks[4];

    // Let consider dims[0] = X, so the shift tells us our left and right neighbours
    MPI_Cart_shift(cart_comm, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);

    // Let consider dims[1] = Y, so the shift tells us our up and down neighbours
    MPI_Cart_shift(cart_comm, 1, 1, &neighbours_ranks[DOWN], &neighbours_ranks[UP]);

    // Get my rank in the new communicator
    int my_rank_top;
    MPI_Comm_rank(cart_comm, &my_rank_top);
    int down = neighbours_ranks[0];
    int up = neighbours_ranks[1];
    int left = neighbours_ranks[2];
    int right = neighbours_ranks[3];

    // Perform computations
    for (int iter = 0; iter < 1; iter++) {
        // Exchange halos
        exchangeHalos(local_matrix, GRID_WIDTH, GRID_HEIGHT, up, down, left, right, cart_comm);

        // Perform the computation
        performComputation(local_matrix, GRID_WIDTH, GRID_HEIGHT);
    }


    MPI_Gather(local_matrix, GRID_WIDTH * GRID_HEIGHT, MPI_DOUBLE, matrix, GRID_WIDTH * GRID_HEIGHT, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the full matrix at the root process
    if (rank == 0) {
        printf("Final grid:\n");
        printMatrix(matrix, GRID_WIDTH, GRID_HEIGHT * size);
        free(matrix);
    }*/


    MPI_Finalize();
    return 0;
}

// This could be done with non-bloquing or unilateral communications as well.
void exchangeHalos(double *local_matrix, int width, int height,
                   int up, int down, int left, int right, MPI_Comm cart_comm) {

    MPI_Status status;

    // Exchange top and bottom rows.
    MPI_Sendrecv(&local_matrix[0], GRID_WIDTH, MPI_DOUBLE, up, 0,
                 &local_matrix[GRID_HEIGHT-1], GRID_WIDTH, MPI_DOUBLE, down, 0,
                 cart_comm, &status);

    MPI_Sendrecv(&local_matrix[GRID_HEIGHT-1], GRID_WIDTH, MPI_DOUBLE, down, 1,
                 &local_matrix[0], GRID_WIDTH, MPI_DOUBLE, up, 1,
                 cart_comm, &status);

    // For left and right, we need to pack and unpack data as they are
    // non-contiguous in memory (or create a new data type).
    double left_column[GRID_HEIGHT];
    double right_column[GRID_HEIGHT];
    double recv_left_column[GRID_HEIGHT];
    double recv_right_column[GRID_HEIGHT];

    for (int i = 0; i < GRID_HEIGHT; i++) {
        left_column[i] = local_matrix[i + 0];
        right_column[i] = local_matrix[i * GRID_WIDTH];
    }

    MPI_Sendrecv(right_column, GRID_HEIGHT, MPI_DOUBLE, right, 2,
                 recv_left_column, GRID_HEIGHT, MPI_DOUBLE, left, 2,
                 cart_comm, &status);

    MPI_Sendrecv(left_column, GRID_HEIGHT, MPI_DOUBLE, left, 3,
                 recv_right_column, GRID_HEIGHT, MPI_DOUBLE, right, 3,
                 cart_comm, &status);

    for (int i = 0; i < GRID_HEIGHT; i++) {
        local_matrix[i] = recv_left_column[i];
        local_matrix[i * GRID_WIDTH-1] = recv_right_column[i];
    }
}

void performComputation(double *local_matrix, int width, int height) {
    // Basic stencil computation: average with immediate neighbors
    double temp_grid[width][height];

    for (int i = 1; i < GRID_HEIGHT-1; i++) {
        for (int j = 1; j < GRID_WIDTH-1; j++) {
            temp_grid[i][j] = (local_matrix[i][j] + local_matrix[i-1][j] + local_matrix[i+1][j]
                               + local_matrix[i][j-1] + local_matrix[i][j+1]) / 5.0;
        }
    }

    // Update the local grid (can be changed to the swap function)
    for (int i = 1; i < GRID_HEIGHT-1; i++) {
        for (int j = 1; j < GRID_WIDTH-1; j++) {
            local_matrix[i][j] = temp_grid[i][j];
        }
    }
}

void printMatrix(double *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%5.2f ", matrix[(i * width) + j]);
        }
        printf("\n");
    }
}

