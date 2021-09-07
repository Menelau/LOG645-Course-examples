/*
 *
 * MPI Send-Receive Hello world example.
 *
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>


int main(int argc, char* argv[]) {
   int message_size = 100;
   char greeting[message_size];  // message buffer
   int comm_size;
   int rank;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (rank != 0) {
      // Creating message message
      sprintf(greeting, "Hello from process %d of %d!",
            rank, comm_size);
      // Send message to process 0
      MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0,
            MPI_COMM_WORLD); 
   } else {  

      for (int p = 1; p < comm_size; p++) {
         // Receive message from process p
         // MPI_Recv(greeting, message_size, MPI_CHAR, p,
         //    MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // MPI_Recv(greeting, message_size, MPI_CHAR, MPI_ANY_SOURCE,
        //     MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         printf("%s\n", greeting);
      } 
   }

   MPI_Finalize(); 
   return 0;
}  
