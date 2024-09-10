#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define FILENAME "monfichier"

int main( int argc, char** argv ){

    int rank, toto = 0;
    MPI_File fh;
    MPI_Status status;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    MPI_File_open( MPI_COMM_WORLD, FILENAME, MPI_MODE_CREATE | MPI_MODE_RDWR,
                   MPI_INFO_NULL, &fh );

    if( 0 == rank ){
        toto = getpid();
        MPI_File_write( fh, &toto, 1, MPI_INT, &status );
        printf( "0 writes %d\n", toto );
    }
    MPI_Barrier( MPI_COMM_WORLD );

    if( 1 == rank ){
        MPI_File_read( fh, &toto, 1, MPI_INT, &status );
        printf( "1 reads %d\n", toto );
    }
        
    MPI_File_close( &fh );
    MPI_Finalize();
    return EXIT_SUCCESS;
}
