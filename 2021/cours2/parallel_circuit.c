/*
 * Circuit Satisfiability MPI version.
 *
 * This Program determines whether a circuit is
 * satisfiable, that is, whether there is a combination of
 * inputs that causes the output of the circuit to be True (1).
 * MPI programm containing the elapsed time and using
 * synchronization functions.
 *
 *  Code adapted from Parallel programming in C with MPI and OpenMP
 *   by Michael J. Quinn (2003)
 */

#include <mpi.h>
#include <stdio.h>

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

int main (int argc, char *argv[]) {
   int count = 0;
   double elapsed_time;
   int global_count;
   int id;
   int p;
   int check_circuit (int, int);

   MPI_Init (&argc, &argv);
   // Makes sure all processes are at this point before starting timer.
   MPI_Barrier (MPI_COMM_WORLD);

   elapsed_time = - MPI_Wtime();

   MPI_Comm_rank (MPI_COMM_WORLD, &id);
   MPI_Comm_size (MPI_COMM_WORLD, &p);

   for (int i = id; i < 65536; i += p){
      count += check_circuit (id, i);
   }

   // Reduce to get global sum. Only process 0 will have this result.
   MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0,
      MPI_COMM_WORLD); 

   // Get total time
   elapsed_time += MPI_Wtime();

   // Using ID 0 for printing
   if (id == 0) {
      printf ("Execution time %8.6f\n", elapsed_time);
      fflush (stdout);
   }
   MPI_Finalize();
   if (id == 0) printf ("There are %d different solutions\n",
      global_count);

   return 0;
}

int check_circuit (int id, int number) {
   int v[16];        /* Each element is a bit of number */
   int i;

   for (int i = 0; i < 16; i++){
      v[i] = EXTRACT_BIT(number,i);
   }
   if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
      && (!v[3] || !v[4]) && (v[4] || !v[5])
      && (v[5] || !v[6]) && (v[5] || v[6])
      && (v[6] || !v[15]) && (v[7] || !v[8])
      && (!v[7] || !v[13]) && (v[8] || v[9])
      && (v[8] || !v[9]) && (!v[9] || !v[10])
      && (v[9] || v[11]) && (v[10] || v[11])
      && (v[12] || v[13]) && (v[13] || !v[14])
      && (v[14] || v[15])) {
         printf ("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
            v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],
            v[10],v[11],v[12],v[13],v[14],v[15]);
         fflush (stdout);
         return 1;

   } else {
      return 0;
   }
}
