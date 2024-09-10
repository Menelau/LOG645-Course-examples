#include <iostream>
#include <cstdlib>
#include <cuda.h>

#define TAILLE 16

__global__ void monkernel( int* A, int N ){
    int x = threadIdx.x;
    if( x < N ) {
        if( 0 == x ){
            A[x] = -1;
        } else {
            A[x] = x;
        }
    }
}

int main(){
    int A[TAILLE];
    int* dA;
    int nbblocks = 4, nbthreads = TAILLE / 4;
    cudaMalloc( (void**)&dA, TAILLE*sizeof( int ) );
    cudaMemset( dA, 0, TAILLE*sizeof( int ) );
    monkernel<<< nbblocks, nbthreads >>>( dA, TAILLE );
    cudaMemcpy( A, dA, TAILLE*sizeof( int ),
                cudaMemcpyDeviceToHost );
    for( auto i = 0 ; i < TAILLE ; i++ ){
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
    return EXIT_SUCCESS;
}
