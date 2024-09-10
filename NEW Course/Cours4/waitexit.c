#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

void toto(){
    printf( "je suis %d, l'enfant de %d\n", getpid(), getppid() );
    sleep( 1 );
    exit( 2 );
}

int main(){
    pid_t cpid,pid;
    int stat;
    pid = fork();
    
    switch( pid ){
    case 0:
        toto();
        break;
    case -1:
        printf( "Erreur lors de la creation de processus -- %d\n", getpid() );
        break;
    default:
        sleep( 2 );
        cpid = wait( &stat );
        printf( "Je suis le processus %d, mon enfant %d a retourné %d\n", getpid(), cpid, WEXITSTATUS( stat ) );
        break;
    }        

    return EXIT_SUCCESS;
}
