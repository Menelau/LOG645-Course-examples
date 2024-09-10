#include <stdlib.h>
#include <stdio.h>  
#include <math.h>

int main( int argc, char** argv ){
  printf("Example 1: defining an array with 10 elements, but manipulating the 11th one.\n");
  double *tab = (double *) malloc(10 * sizeof(double));
  double aa[2];
  aa[0] = 10.0;
  tab[10] = 25.2;       //No out of bounds error...
  printf( "%lf\n", aa[0] );
  printf( "%lf\n", sqrt(tab[10]) ); //doing operation in this memory address.

  printf("Example 2: defining an array with 10 elements, but manipulating the 1000th one.\n");
  int idx = 10000000;
  double *tab2 = (double *) malloc(10 * sizeof(double));
  double aa2[2];
  aa2[0] = 10.0;
  tab2[idx] = 25.2;
  printf( "%lf\n", aa2[0] );
  printf( "%lf\n", sqrt(tab2[idx]) );
  
  return 0;
}
