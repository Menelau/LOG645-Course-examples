#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


//OpenCL kernel as a string
const char *matMulKernel =
"__kernel                                                     \n"
"void matMul_Kernel(__global int *A,                          \n"
"                   __global int *B,                          \n"
"                   __global int *C, int N)                   \n"
"{                                                            \n"
"    //Get the index of the work-item                         \n"
"    int col = get_global_id(0);                              \n"
"    int row = get_global_id(1);                              \n"
"    C[row * N + col] = 0;                                    \n"
"    for (int i = 0; i < N; i++) {                            \n"
"        C[row * N + col] += A[row * N + i] * B[i * N + col]; \n"
"  }                                                          \n"
"}                                                            \n";

int verifyResults(int* matrix_A, int* matrix_B, int* matrix_C, int N);

int main(void) {

  // Allocate space for matrices A, B and C
  int N = 2048;
  size_t bytes = N * N * sizeof(int);
  int *h_matrixA = (int*) malloc(bytes);
  int *h_matrixB = (int*) malloc(bytes);
  int *h_matrixC = (int*) malloc(bytes);

  for(int i = 0; i < N * N; i++){
    h_matrixA[i] = rand() % 100;
    h_matrixB[i] = rand() % 100;
    h_matrixC[i] = -1;
  }

  // Well all OpenCL boilerplate code that we can just copied from previous examples...
  cl_platform_id * platforms = NULL;
  cl_device_id *device_list = NULL;

  cl_uint num_platforms;
  cl_context context;
  cl_command_queue command_queue;
  cl_program matMulProgram;
  cl_kernel kernel;

  // 1) Get platform and device information
  cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

  // 2) Get the devices list and choose the device you want to run on
  cl_uint num_devices;
  clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
  clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

  // 3) Create one OpenCL context for each device in the platform
  context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

  // 4) Create a command queue
  command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

  // 5) Allocated memory on the device (for each array)
  cl_mem d_matrixA = clCreateBuffer(context, CL_MEM_READ_ONLY,
            bytes, NULL, &clStatus);
  cl_mem d_matrixB = clCreateBuffer(context, CL_MEM_READ_ONLY,
            bytes, NULL, &clStatus);
  cl_mem d_matrixC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            bytes, NULL, &clStatus);

  // 6) Copy data from host to device
  clStatus = clEnqueueWriteBuffer(command_queue, d_matrixA, CL_TRUE, 0,
             bytes, h_matrixA, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, d_matrixB, CL_TRUE, 0,
             bytes, h_matrixB, 0, NULL, NULL);

  // 7) Create a program from the kernel source
  matMulProgram = clCreateProgramWithSource(context, 1,
            (const char **)&matMulKernel, NULL, &clStatus);

  // 8) Build the program
  clStatus = clBuildProgram(matMulProgram, 1, device_list, NULL, NULL, NULL);

  // // 9) Create the OpenCL kernel
  kernel = clCreateKernel(matMulProgram, "matMul_Kernel", &clStatus);

  // 9.1) Set the arguments of the kernel

 clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_matrixA);
 clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_matrixB);
 clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_matrixC);
 clStatus = clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);


  // SETUP KERNEL EXECUTION BY GLOBAL AND LOCAL SIZES
  // M * M elements to compute
  const size_t global_size[] = {N, N};
  // 256 threads for each unity (NVIDIA LIKE)
  const size_t local_size[] = {16, 16};
  int problemDimension = 2;
  clStatus = clEnqueueNDRangeKernel(command_queue, kernel, problemDimension, NULL,
             global_size, local_size, 0, NULL, NULL);

  // 11) Transfer data Device to Host
  clStatus = clEnqueueReadBuffer(command_queue, d_matrixC, CL_TRUE, 0,
             bytes, h_matrixC, 0, NULL, NULL);

  // Wait for all the comands to complete.
  clStatus = clFlush(command_queue);
  clStatus = clFinish(command_queue);

  verifyResults(h_matrixA, h_matrixB, h_matrixC, N);

  // 12) Release allocated memory on devices
  clStatus = clReleaseMemObject(d_matrixA);
  clStatus = clReleaseMemObject(d_matrixB);
  clStatus = clReleaseMemObject(d_matrixC);


  // 13)Finally release all OpenCL allocated objects.
  clStatus = clReleaseKernel(kernel);
  clStatus = clReleaseProgram(matMulProgram);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);

  free(h_matrixA);
  free(h_matrixB);
  free(h_matrixC);
  free(platforms);
  free(device_list);

  return 0;
}


// Functon to compare the GPU results with the calculation performed in the CPU.
int verifyResults(int* matrix_A, int* matrix_B, int* matrix_C, int N) {
    long correct = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;     //not a perfect nested loop :-(
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += matrix_A[i * N + k] * matrix_B[k * N + j];
            }
            // Check against the CPU result
            if (tmp == matrix_C[i * N + j]) correct++;
        }
    }
    printf("%ld of %ld values verified!", correct, (long)N * N);
    return 0;
}
