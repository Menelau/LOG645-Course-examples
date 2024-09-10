#include <stdio.h>
#include <stdlib.h>
#include <omp.h>   // including OpenMP library

void Hello(void);  // Thread function

int main(int argc, char* argv[]) {

/* by default the compilers use the number of cores as the number of threads.
 We can specify a different number using the directive "num_threads(thread_count)" */
#  pragma omp parallel
   Hello();
   printf("end");
   return 0; 
}


void Hello(void) {
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();

   printf("Hello I'm thread %d of %d\n", my_rank, thread_count);

}
