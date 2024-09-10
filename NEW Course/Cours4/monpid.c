#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main(){
    pid_t pid;
    pid = getpid();
    printf( "Je suis le processus %d\n", pid );
    return EXIT_SUCCESS;
}
