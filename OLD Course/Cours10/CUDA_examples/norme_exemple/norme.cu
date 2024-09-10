#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TAILLE 23
#define NBTHREADS 4
#define RAND( a, b ) a + ( ( b - a + 1 ) * (double)rand() ) / (double) RAND_MAX

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


__global__ void norme_v1( double* data, double* res ){
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    for( int i = x ; i < TAILLE ; i += blockDim.x )  {
        res[threadIdx.x] += ( data[i] * data[i] );
    }

}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
#warning "attention, atomicAdd n'est pas dispo"
__device__ double atomicAdd(double* a, double b) { *a += b; return b; }
#endif

__global__ void norme_v2( double* data, double* res ){
    __shared__ double sum;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    /* Initialisation de la norme */
    if( 0 == threadIdx.x ) sum = 0.0;
    __syncthreads();

    double tmp = 0.0;
    for( int i = x ; i < TAILLE ; i += blockDim.x )  {
        tmp += ( data[i] * data[i] );
    }

    /* On calcule la somme globale et on met dans la variable de sortie */
    tmp = atomicAdd( &sum, tmp );
    
    __syncthreads();
    if( 0 == threadIdx.x ) *res = sqrt( sum );
}

__global__ void norme_v3( double* data, double* res ){
    __shared__ double sum;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y, offset;
    if( TAILLE * NBTHREADS == 0 ){
        y = TAILLE / NBTHREADS;
        offset = x*y;
    } else {
        y = (int) floor( (double) TAILLE / (double) ( NBTHREADS - 1 ));
        offset = x*y;
        if( x+1 == NBTHREADS ) {
            y = TAILLE % ( NBTHREADS - 1 );
        } 
    }
    //    printf( "%d y: %d \n", threadIdx.x, y );
    
    /* Initialisation de la norme */
    if( 0 == threadIdx.x ) sum = 0.0;
    __syncthreads();

    double tmp = 0.0;
    for( int i = 0 ; i < y ; i++ )  {
        tmp += ( data[ offset + i] * data[ offset + i] );
    }

    /* On calcule la somme globale et on met dans la variable de sortie */
    tmp = atomicAdd( &sum, tmp );
    
    __syncthreads();
    if( 0 == threadIdx.x ) *res = sqrt( sum );
}

int main( void ) {

    double A[TAILLE];
    double* d_A;
    double n, *nn;

    checkCudaErrors( cudaMalloc( (void**) &d_A, TAILLE*sizeof( double ) ) );

    srand( 0 ); // pour la reproductibilité
    for( int i = 0 ; i < TAILLE ; i++ ) A[i] = RAND( -100, 100 );
    printf( "Init: \n" );
    for( int i = 0 ; i < TAILLE ; i++ ) printf( "%.2lf  ", A[i] );
    printf( "\n" ); 

    cudaMemcpy( d_A, A, TAILLE*sizeof( double ), cudaMemcpyHostToDevice ) ;

    cudaMalloc( (void**) &nn, sizeof( double ) );

    int nbthreads = 4;
    double res_thr[nbthreads];
    double* p_res_thr;

    cudaMalloc( (void**) &p_res_thr, NBTHREADS*sizeof( double ) );
    cudaMemset( p_res_thr, 0, NBTHREADS*sizeof( double ) );
    
    norme_v1<<<1, NBTHREADS>>>( d_A, p_res_thr );

    cudaMemcpy( res_thr, p_res_thr, NBTHREADS*sizeof( double ), cudaMemcpyDeviceToHost ) ;
    cudaFree( p_res_thr );
    n = 0.0;
    for( int i = 0 ; i < NBTHREADS ; i++ ){
        n += res_thr[i];
    }
    n = sqrt( n );

    printf( "Norm v1: %.2lf\n", n );
    
    norme_v2<<<1, NBTHREADS>>>( d_A, nn );
    
    cudaMemcpy( &n, nn, sizeof( double ), cudaMemcpyDeviceToHost ) ;
    printf( "Norm v2: %.2lf\n", n );    

    norme_v3<<<1, NBTHREADS>>>( d_A, nn );
    
    cudaMemcpy( &n, nn, sizeof( double ), cudaMemcpyDeviceToHost ) ;
    printf( "Norm v3: %.2lf\n", n );    
    
    /* On verifie le calcul */
    n = 0.0;
    for( int i = 0 ; i < TAILLE ; i++ ) n += A[i]*A[i];
    n = sqrt( n );
    printf( "Norm   : %.2lf\n", n );    
        
    cudaFree( d_A );
    return EXIT_SUCCESS;
}
