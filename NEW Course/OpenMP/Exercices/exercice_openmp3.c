#include <stdio.h>
#include <stdlib.h>

int main(void){

    int bufferSize = 256;
    int iterationCount = 100;
    int currentIteration = 0;
    char** buffers = (char**)(malloc(iterationCount * sizeof(char*)));

    #pragma omp parallel for
    for (int i = 0; i < iterationCount; i++){
        char* buffer = (char*)(malloc((bufferSize + 1) * sizeof(char)));
        int currentIterationLocal = i;
        currentIteration = i;
        int writtenChars = snprintf(buffer, bufferSize, "i = #%d, ", i);
        writtenChars += snprintf(&buffer[writtenChars], bufferSize - writtenChars, "currentIteration = %d, ", currentIteration);
        writtenChars += snprintf(&buffer[writtenChars], bufferSize - writtenChars, "iterationCount = %d, ", iterationCount);
        writtenChars += snprintf(&buffer[writtenChars], bufferSize - writtenChars, "currentIterationLocal = %d \n", currentIterationLocal);
        buffer[writtenChars] = '\n';
        buffers[i] = buffer;
    }

    for (int i = 0; i < iterationCount; i++){
        printf(buffers[i]);
        free(buffers[i]);
    }
    free(buffers);
    system("pause");
    return 0;
}
