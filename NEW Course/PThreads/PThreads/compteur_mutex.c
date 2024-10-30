#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NB_ENFANTS 8

static unsigned int cnt = 0;
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

void* enfant( void* arg ){
    unsigned int* k = (unsigned int*) arg;
    int ex = 0;
    printf( "Je suis l'enfant %d\n", *k );
    /* Attention à l'exclusion mutuelle */
    pthread_mutex_lock( &mtx );
    cnt++;
    pthread_mutex_unlock( &mtx );
    pthread_exit( (void*)&(ex) );
}

int main(){
    pthread_t t[NB_ENFANTS];
    unsigned int i;
    int* ret;

    for( i = 0 ; i < NB_ENFANTS ; i++ ){
        pthread_create( &(t[i]), NULL, &enfant, (void*) &i );
    }

    for( i = 0 ; i < NB_ENFANTS ; i++ ){
        pthread_join( t[i], (void**)&ret );
    }
    printf( "Valeur du compteur: %d\n", cnt );

    return EXIT_SUCCESS;
}
