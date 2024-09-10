#include <stdio.h>
#include <stdlib.h>

int fib( int n ){
    int i, j;
    if( n < 2 ) return n;
    else {
        i = fib( n - 1 );
        j = fib( n - 2 );
        return i + j;
    }
}

int main( int argc, char** argv ){
    int N = 100000;
    if( argc > 1 ){
        N = atoi( argv[1] );
    }
    int res = fib( N );
    printf( "fib( %d ) = %d\n", N, res );
    return EXIT_SUCCESS;
}
