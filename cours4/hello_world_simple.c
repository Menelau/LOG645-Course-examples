#include <stdio.h>
#include <stdlib.h>

int main(void) {

/* by default the compilers use the number of cores as the number of threads.
 We can specify a different number using the directive "num_threads(thread_count)" */
#  pragma omp parallel num_threads(42)
   {
      printf("Hello world, I'm a thread!\n");
   }

}
