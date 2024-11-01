#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>

int main(){
    pid_t pid, cpid;

    // have the calendar as a progam to be executed.
    char* command = "/usr/bin/cal";
    char* args[3] = { "/usr/bin/cal", NULL };

    pid = fork();
    switch( pid ){
    //processus enfant
    case 0:
        printf( "Je suis le processus enfant %d\n", getpid() );
        execv( command, args );
        perror( "execv returned" );         /* ne devrait jamais retourner */
        break;
    case -1:
        printf( "Erreur lors de la creation de processus -- %d\n", getpid() );
        break;
    //processus parent
    default:
        printf( "Je suis le processus parent %d et je viens de lancer mon enfant %d\n", getpid(), pid );
        cpid = wait( NULL );
        break;
    }        
    return EXIT_SUCCESS;
}
