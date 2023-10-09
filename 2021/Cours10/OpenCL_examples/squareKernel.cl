__kernel void powKernel(__global float* Inoout __global, float* output){
    
    int index = get_global_id(0);
    I[index] = I[index] * I[index];

}
