/*
 ============================================================================
 Name        : LOG645_H15_C06_PT_3.c
 Author      : Carlos Vazquez, Modified by Rafael M. O. Cruz
 Version     : 2.0
 Copyright   : Your copyright notice
 Description : Nested threads in C, Ansi-style
 ============================================================================
 */

#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void afficher(int n, char lettre){
    int i,j;
    for (j=1; j<n; j++){
        for (i=1; i < 10000000; i++);
        printf("%c",lettre);
        fflush(stdout);
    }
}

void *threadA(void *inutilise){
    afficher(100,'A');
    printf("\n End thread A\n");
    fflush(stdout);
    pthread_exit(NULL);
}
void *threadC(void *inutilise){
    afficher(150,'C');
    printf("\n End thread C\n");
    fflush(stdout);
    pthread_exit(NULL);
}

void *threadB(void *inutilise){
    pthread_t thC;
    pthread_create(&thC, NULL, threadC, NULL);
    afficher(50,'B');
    printf("\n Thread B wait for thread C to finish\n");
    pthread_join(thC,NULL);
    printf("\n End thread B\n");
    fflush(stdout);
    pthread_exit(NULL);
}

int main(){
    int i;
    pthread_t thA, thB;

    printf("Creating Thread A\n");
    pthread_create(&thA, NULL, threadA, NULL);
    printf("Creating Thread B\n");
    pthread_create(&thB, NULL, threadB, NULL);

    sleep(50);

    //wait for threads A and B to finish
    printf("Joining threads A and B\n");
    pthread_join(thA,NULL);
    pthread_join(thB,NULL);

    return 0;
}
