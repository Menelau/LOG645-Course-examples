/*
 *   Circuit Satisfiability, Serial version
 *
 *   This Program determines whether a circuit is
 *   satisfiable, that is, whether there is a combination of
 *   inputs that causes the output of the circuit to be True (1).
 *
 *  Code adapted from Parallel programming in C with MPI and OpenMP
 *   by Michael J. Quinn
 */

#include <stdio.h>

// Return 1 if 'i'th bit of 'n' is 1; 0 otherwise
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

int main (void) {
   int check_circuit (int);
   int count = 0;

   for (int i = 0; i < 65536; i++){
      count += check_circuit (i);
   }
   printf("Number of circuits : %d", count);
   return 0;
}


int check_circuit (int number) {
   int v[16];        /* Each element is a bit of number */

   for (int bit = 0; bit < 16; bit++){
      v[bit] = EXTRACT_BIT(number, bit);    
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
      printf ("%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n",
         v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],
         v[10],v[11],v[12],v[13],v[14],v[15]);
      fflush (stdout);
      return 1;
   } else {
      return 0;
   }
}
