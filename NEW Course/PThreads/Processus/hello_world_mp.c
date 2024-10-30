#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
    pid_t pid;
    pid = getpid();
    printf("Starting program... process id = %d\n", pid);
    fflush(stdout);
    int id = fork();
    printf("Hello from process %d\n", id);
    fflush(stdout);


}
