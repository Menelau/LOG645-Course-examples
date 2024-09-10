#include <stdio.h>
#include <stdlib.h>

// This example uses SSE. To make life a bit easier (there are multiple SSE files)
// we are adding Adding all intrinsics with this include.
#include <x86intrin.h>

/*
  Code to compile
  gcc -o load_store load_store.c -msse
*/

int main( void ){

    int in[8]= {8,7,6,5,4,3,2,1};
    int out[8];
    __m128i vec;

    // look that we use the _si modifier and i in general.
    vec = _mm_loadu_si128( (__m128i*) in );

    // storing the element
    _mm_storeu_si128( (__m128i*) out, vec );

    //only the first part of the array was collected and stored in out.
    printf( "Out: " );
    for(int i = 0 ; i < 8 ; i++ ){
        printf( "%i  ", out[i] );
    }
    printf( "\n" );
    getchar();
    //second part: passing a different memory address (starting at idx=4)
    vec = _mm_loadu_si128( (__m128i*) &in[4] );
    _mm_storeu_si128( (__m128i*) &out[4], vec );

    //both parts of the array were copied now.
    printf( "Out: " );
    for(int i = 0 ; i < 8 ; i++ ){
        printf( "%i  ", out[i] );
    }
    printf( "\n" );

}
