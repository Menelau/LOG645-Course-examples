// Kernel to perform a simple matrix multiplication with OpenCL. We will optimize it later...

__kernel void matrixMultiplicationKernel(__global int *matrix_A,
                                         __global int *matrix_B,
                                         __global int *matrix_C, int N) {
  // Compute each thread's x and y indexes
  int col = get_global_id(0);
  int row = get_global_id(1);
  // Iterate over row, and column
  matrix_C[row * N + col] = 0;
  for (int i = 0; i < N; i++) {
    // Accumulate results for a single element in the output matrix
    matrix_C[row * N + col] += matrix_A[row * N + i] * matrix_B[i * N + col];
  }
}
