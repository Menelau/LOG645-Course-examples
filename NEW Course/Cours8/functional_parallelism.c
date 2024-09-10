/*
This code example shows how to exploit functional parallelism using "OpenMP sections" directives.

Note: OpenMP offers two directions "sections" plural and "section" singular. They have different behavior.
*/

#include <stdio.h>
#include <stdlib.h>
#include "math.h"
// trying to be cross platform here...
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

float alpha(void){
    //sleep and return random float;
    int random_sleep = rand() % 10;
    float max_float = 5.0;
    sleep(random_sleep);
    float v = ((rand()/(float)(RAND_MAX)) * max_float);
    return v;
}

float beta(void){
    //sleep and return random float;
    int random_sleep = rand() % 10;
    float max_float = 10.0;
    sleep(random_sleep);
    float w = (float) ((rand()/(float)(RAND_MAX)) * max_float);
    return w;
}

float delta(void){
    //sleep and return random float;
    int random_sleep = rand() % 10;
    float max_float = 1.0;
    sleep(random_sleep);
    float y = (float) ((rand()/(float)(RAND_MAX)) * max_float);
    return y;
}

float gamma(float x, float w){
    float result = (x * 100.0) + pow(w, 2.0);
    return result;
}

float epsilon(float x, float y){
    float result = x + y;
    return result;
}

float v, w, y, x;

int main(int argc, char** argv){

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            v = alpha();
            printf("v is %6.2f \n", v);

        }

        #pragma omp section
        {
            w = beta();
            printf("w is %6.2f \n", w);

        }

        #pragma omp section
        {
            y = delta();
            printf("y is %6.2f \n", y);

        }
    }// Implicit barrier at the end of "sections" directive. All threads must finish.

    float x = gamma(v, w);
    float result = epsilon(x, y);
    printf("Result is: %6.2f \n", result);
    return 0;
}

