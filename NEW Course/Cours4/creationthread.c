#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

struct mastruct {
    int a;
    char b;
};

void* fonctionEnfant( void* arg ){
    struct mastruct* s_arg = (struct mastruct*) arg;
    printf( "Arguments: %d %c\n", s_arg->a, s_arg->b );
    pthread_exit( (void*)&(s_arg->a) );
}

int main(){
    pthread_t t;
    struct mastruct s;
    int* ret;
    s.a = 120;
    s.b = 'l';

    pthread_create( &t, NULL, &fonctionEnfant, (void*)&s );
    pthread_join( t, (void**)&ret );
    printf( "Returned value: %d\n", *ret );

    return EXIT_SUCCESS;
}
