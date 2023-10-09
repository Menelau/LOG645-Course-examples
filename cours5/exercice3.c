#include <stdio.h>

int main(void){
	
	int total = 0;
	#pragma omp parallel for reduction(+:total)
	for (int i = 0; i < 100; i++){
		total += i;
	}
	printf("total : %d", total);

}
