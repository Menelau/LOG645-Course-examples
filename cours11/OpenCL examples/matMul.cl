__kernel
void matMul_Kernel(__global int *A,
                   __global int *B,
                   __global int *C, int N)
{
    //Get the index of the work-item
    int col = get_global_id(0);
    int row = get_global_id(1);
    C[row * N + col] = 0;
    for (int i = 0; i < N; i++) {
        C[row * N + col] += A[row * N + i] * B[i * N + col];
  }
}
