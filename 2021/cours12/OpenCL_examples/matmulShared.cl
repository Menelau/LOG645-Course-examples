__kernel void tiledMatmulKernel(__global int *marix_A,
                                __global int *matrix_B,
                                __global int *result) {
    int size = get_global_size(0);
    const int l_i = get_local_id(0);
    const int l_j = get_local_id(1);
    const int g_i = get_global_id(0);
    const int g_j = get_global_id(1);
 
    // Local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
    __local int Asub[TILE_SIZE][TILE_SIZE];
    __local int Bsub[TILE_SIZE][TILE_SIZE];
 
    int acc = 0;
    int t_i;
    int t_j;
    const int numTiles = size/TILE_SIZE;

    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        t_i = t*TILE_SIZE + l_i;
        t_j = t*TILE_SIZE + l_j;

        Asub[l_i][l_j] = A[g_j*size + t_i];
        Bsub[l_i][l_j] = B[t_j*size + g_i];
 
        // synchronization barrier. shared memory must be complete before
        // proceeding with the calculations
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += Asub[k][l_j] * Bsub[l_i][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    result[g_j*size + g_i] = acc;
   
}
