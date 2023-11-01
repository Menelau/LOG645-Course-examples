#include <stdio.h>
#include <stdlib.h>

#define DEFAULTN 100

int fib( int n ){
    int i, j;
    if( n < 2 ) return n;
    else {
#pragma omp task shared( i ) firstprivate( n )
        i = fib( n - 1 );
#pragma omp task shared( j ) firstprivate( n )
        j = fib( n - 2 );
#pragma omp taskwait
        return i + j;
    }
}

int main( int argc, char** argv ){

    int res, N = DEFAULTN;
    if( argc > 1 ){
        N = atoi( argv[1] );
    }
#pragma omp parallel shared( N )
    {
#pragma omp single
        res = fib( N );
    }
    printf( "fib( %d ) = %d\n", N, res );
    
    return EXIT_SUCCESS;
}
