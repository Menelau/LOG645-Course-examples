#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>

#define BUFFSIZE 16

void enfant( int* pipefd ){
    char message[BUFFSIZE];
    int rc;

    close( pipefd[1] ); /* Je ne vais pas ecrire ici */

    printf( "Je suis le processus enfant %d\n", getpid() );
    rc = read( pipefd[0], message, BUFFSIZE );
    printf( "ENFANT lu %d octets: %s\n", rc, message );
    rc = read( pipefd[0], message, BUFFSIZE );
    if( 0 == rc ){
        printf( "Le tube a été fermé par l'autre processus\n" );
        close( pipefd[0] ); /* Je ferme mon bout en lecture */
    }
}

void parent( int* pipefd ){
    char* envoi = "Hello";
    
    close( pipefd[0] ); /* Je ferme mon bout en lecture */
    
    printf( "Je suis le processus parent %d\n", getpid() );
    write( pipefd[1], envoi, sizeof( envoi ) );
    close( pipefd[1] ); /* Je commence par fermer */
}

int main(){

    pid_t pid, cpid;
    int pipefd[2];
    int rc;
    
    /* Creation du tube */
    rc = pipe( pipefd );
    if( rc < 0 ){
        perror( "Erreur creation du tube\n" );
        return EXIT_FAILURE;
    }

    pid = fork();
    switch( pid ){
    case 0:
        enfant( pipefd );
        break;
    case -1:
        printf( "Erreur lors de la creation de processus -- %d\n", getpid() );
        break;
    default:
        parent( pipefd );
        cpid = wait( NULL );
        printf( "Mon enfant %d a fini\n", cpid );
        break;
    }        

    return EXIT_SUCCESS;
}
