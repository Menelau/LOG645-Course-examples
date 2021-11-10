#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define VECTOR_SIZE 1024


//OpenCL kernel as a string
const char *saxpyKernel =
"__kernel                                   \n"
"void saxpy_kernel(float alpha,     \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
" printf(\"hey hey hey\");                  \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";


int main(void) {

  // Allocate space for vectors A, B and C
  float alpha = 2.0;
  size_t bytes = VECTOR_SIZE * sizeof(float);
  float *h_A = (float*) malloc(bytes);
  float *h_B = (float*) malloc(bytes);
  float *h_C = (float*) malloc(bytes);

  for(int i = 0; i < VECTOR_SIZE; i++){
    h_A[i] = i;
    h_B[i] = VECTOR_SIZE - i;
  }

  // 0) Create OpenCL Structures
  cl_platform_id * platforms = NULL;
  cl_device_id *device_list = NULL;

  cl_uint num_platforms;
  cl_context context;
  cl_command_queue command_queue;
  cl_program saxpy_program;
  cl_kernel saxpy_kernel;

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
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
            bytes, NULL, &clStatus);
  cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,
            bytes, NULL, &clStatus);
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            bytes, NULL, &clStatus);

  // 6) Copy data from host to device
  clStatus = clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0,
             bytes, h_A, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0,
             bytes, h_B, 0, NULL, NULL);

  // 7) Create a program from the kernel source
  saxpy_program = clCreateProgramWithSource(context, 1,
            (const char **)&saxpyKernel, NULL, &clStatus);

  // 8) Build the program
  clStatus = clBuildProgram(saxpy_program, 1, device_list, NULL, NULL, NULL);

  // // 9) Create the OpenCL kernel
  saxpy_kernel = clCreateKernel(saxpy_program, "saxpy_kernel", &clStatus);

  // 9.1) Set the arguments of the kernel
  clStatus = clSetKernelArg(saxpy_kernel, 0, sizeof(float), (void *)&alpha);
  clStatus = clSetKernelArg(saxpy_kernel, 1, sizeof(cl_mem), (void *)&d_A);
  clStatus = clSetKernelArg(saxpy_kernel, 2, sizeof(cl_mem), (void *)&d_B);
  clStatus = clSetKernelArg(saxpy_kernel, 3, sizeof(cl_mem), (void *)&d_C);

  // 10) Execute the OpenCL kernel on the list
  size_t global_size = VECTOR_SIZE; // Process the entire lists
  size_t local_size = 64;           // Process one item at a time
  clStatus = clEnqueueNDRangeKernel(command_queue, saxpy_kernel, 1, NULL,
             &global_size, &local_size, 0, NULL, NULL);

  // 11) Transfer data Device to Host
  clStatus = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0,
             bytes, h_C, 0, NULL, NULL);

  // Wait for all the comands to complete.
  clStatus = clFinish(command_queue);

  // 12) Release allocated memory on devices
  clStatus = clReleaseMemObject(d_A);
  clStatus = clReleaseMemObject(d_B);
  clStatus = clReleaseMemObject(d_C);

  // Display the result to the screen
for(int i = 0; i < VECTOR_SIZE; i++){
    printf("%f * %f + %f = %f\n", alpha, h_A[i], h_B[i], h_C[i]);
}
  // 13)Finally release all OpenCL allocated objects.
  clStatus = clReleaseKernel(saxpy_kernel);
  clStatus = clReleaseProgram(saxpy_program);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);

  free(h_A);
  free(h_B);
  free(h_C);
  free(platforms);
  free(device_list);

  return 0;
}
