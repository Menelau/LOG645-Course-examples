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
