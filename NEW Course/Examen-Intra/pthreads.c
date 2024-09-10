//Parmi les segments de code suivants, décrivant la fonction Send_msg, seulement un fonctionne correctement. Lequel?

//L'initialisation est donnée par le segment :


   thread_handles = malloc (thread_count*sizeof(pthread_t));

   messages = malloc(thread_count*sizeof(char*));

   semaphores = malloc(thread_count*sizeof(sem_t));

   for (thread = 0; thread < thread_count; thread++) {

      messages[thread] = NULL;

      /* Initialize all semaphores to 0 -- i.e., locked */

      sem_init(&semaphores[thread], 0, 0);

   }

   for (thread = 0; thread < thread_count; thread++)

      pthread_create(&thread_handles[thread], (pthread_attr_t*) NULL, Send_msg, (void*) thread);

   for (thread = 0; thread < thread_count; thread++) {

      pthread_join(thread_handles[thread], NULL);

   }

   for (thread = 0; thread < thread_count; thread++) {

      free(messages[thread]);

      sem_destroy(&semaphores[thread]);

   }

   free(messages);

   free(semaphores);

   free(thread_handles);

void *Send_msg(void* rank) {

   long my_rank = (long) rank;

   long dest = (my_rank + 1) % thread_count;

   long source = (my_rank - 1 + thread_count) % thread_count;

   char* my_msg = malloc(MSG_MAX * sizeof(char));

   sprintf(my_msg, "Hello to %ld from %ld", dest, my_rank);

   messages[dest] = my_msg;

   /* Notifier le thread de destination qu’il peut continuer */

   sem_post(sems[dest]);

   /* Attendre par le thread d’origine du message */

   sem_wait(sems[my_rank]);

   if (messages[my_rank] != NULL)

      printf("Thread %ld > %s\n", my_rank, messages[my_rank]);

   else

      printf("Thread %ld > No message from %ld\n", my_rank, source);

   return NULL;

}  /* hello */
