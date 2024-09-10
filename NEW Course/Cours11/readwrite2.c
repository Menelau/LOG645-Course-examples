#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define FILENAME "monfichier"

int main( int argc, char** argv ){

    int rank, size, toto = 0;
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    MPI_File_open( MPI_COMM_WORLD, FILENAME, MPI_MODE_CREATE | MPI_MODE_RDWR,
                   MPI_INFO_NULL, &fh );

    toto = getpid();
    offset = rank * sizeof( int );
    MPI_File_write_at( fh, offset, &toto, 1, MPI_INT, &status );
    printf( "%d writes %d\n", rank, toto );
    
    MPI_Barrier( MPI_COMM_WORLD );

    offset = ( ( rank + 1 ) % size )  * sizeof( int );
    MPI_File_read_at( fh, offset, &toto, 1, MPI_INT, &status );
    printf( "%d reads %d\n", rank, toto );
        
    MPI_File_close( &fh );
    MPI_Finalize();
    return EXIT_SUCCESS;
}
