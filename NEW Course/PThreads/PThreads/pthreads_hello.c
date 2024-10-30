/*
 * Simple Hello World code using PThreads.
 *
 *
 * Run: ./pthreads_hello <number of threads>
 *
 * Code inspired by Introduction to Parallel Programming by Peter Pacheco.
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 

/* Global variable:  accessible to all threads */
int thread_count;  

void Usage(char* prog_name);
void *Hello(void* rank);  /* Thread function */

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   long       thread;  /* Use long in case of a 64-bit system */
   pthread_t* thread_handles; 

   /* Get number of threads from command line */
   if (argc != 2){
         fprintf(stderr, "usage: %s <number of threads>\n", argv[0]);
   }
   // convert string to long
   thread_count = strtol(argv[1], NULL, 10);  

   if (thread_count <= 0){
         fprintf(stderr, "Thread number cannot be negative\n");
   }

   // allocated memory for all threads
   thread_handles = malloc(thread_count*sizeof(pthread_t));

   //create threads
   for (thread = 0; thread < thread_count; thread++)  
      pthread_create(&thread_handles[thread], NULL,
          Hello, (void*) thread);  

   printf("Hello from the main thread\n");

   // wait for all threads to finish
   for (thread = 0; thread < thread_count; thread++) 
      pthread_join(thread_handles[thread], NULL); 

   free(thread_handles);
   return 0;
}

/*
   Hello world function.
*/
void *Hello(void* rank) {
   long my_rank = (long) rank;

   printf("Hello from thread %ld of %d\n", my_rank, thread_count);
   return NULL;
}

