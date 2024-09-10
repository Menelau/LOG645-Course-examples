#include <stdio.h>
#include <omp.h>

int main(void){
    int i = 0;
    #pragma omp parallel for num_threads(4) \
     schedule( static, 10)
    for (int i = 0; i < 100; i++) {
        printf("Thread id %d running iteratio %d\n",
            omp_get_thread_num(), i);
    }
    return 0;
}
