#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define FILENAME "monfichier"
#define TAILLE 1

int main( int argc, char** argv ){

    int rank, size, toto[2*TAILLE];
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset;
    MPI_Datatype mytype, typeecrit;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    MPI_File_open( MPI_COMM_WORLD, FILENAME, MPI_MODE_CREATE | MPI_MODE_RDWR,
                   MPI_INFO_NULL, &fh );

    /* Création d'un datatype qui contient deux entiers, séparés par le nombre de processus */
    MPI_Type_vector( 2, TAILLE, size, MPI_INT, &mytype );
    MPI_Type_commit( &mytype );

    /* On calcule un offset */

    offset = (MPI_Offset)rank * TAILLE * sizeof( int );
    MPI_File_set_view( fh, offset, MPI_INT, mytype, 
                       "native", MPI_INFO_NULL );

    /* On crée un datatype pour ce qu'on va écrire dans le fichier */
   MPI_Type_contiguous( TAILLE, MPI_INT, &typeecrit );
   MPI_Type_commit( &typeecrit );

   toto[0] = rank;
   toto[1] = getpid();

   /* On ecrit */
   MPI_File_write( fh, toto, 2, typeecrit, &status );   
   printf( "%d a ecrit %d - %d\n", rank, toto[0], toto[1] );

   MPI_Barrier( MPI_COMM_WORLD );

   /* On lit */
   offset = (MPI_Offset)( ( rank + 1 ) % size ) * TAILLE * sizeof( int );
   MPI_File_set_view( fh, offset, MPI_INT, mytype, 
                      "native", MPI_INFO_NULL );
   MPI_File_read( fh, toto, 2, typeecrit, &status );   

   printf( "%d a lu %d - %d\n", rank, toto[0], toto[1] );
   
   
   MPI_File_close( &fh );
   MPI_Type_free( &mytype );
   MPI_Type_free( &typeecrit );
   MPI_Finalize();
   return EXIT_SUCCESS;
}
