#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
    printf( "Bonjour 1 %d\n", getpid() );
    fork();
    printf( "Bonjour 2 %d\n", getpid() );
    fork();
    printf( "Bonjour 3 %d\n", getpid() );
    fork();
    printf( "Bonjour 4 %d\n", getpid() );

    return EXIT_SUCCESS;
}
