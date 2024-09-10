
#include <stdio.h>

int main(void){

	#pragma omp parallel for
	for (int i = 0; i < 100; ++i){
		#pragma omp critical{

		printf("Je suis #%d\n", i);
		}
	}
	return 0;
}
