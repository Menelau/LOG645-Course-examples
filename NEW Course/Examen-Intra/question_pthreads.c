#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
int c = 0;

void foo(){
    pthread_mutex_lock( &mut );
    c++;
    pthread_mutex_unlock( &mut );
}

int main(){
    pthread_t t[2];

    pthread_create( &t[0], NULL, &foo, NULL );
    pthread_create( &t[1], NULL, &foo, NULL );

    pthread_join( &t[0], NULL );
    pthread_join( &t[1], NULL );
    printf( "%d\n", c );
    
    return EXIT_SUCCESS;
}
