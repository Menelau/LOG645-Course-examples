#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {

	int rang;
	int nombreProc;
	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	MPI_Comm_size(MPI_COMM_WORLD, &nombreProc);
	int matrice[nombreProc];

	if (rang == 0){

		// matrice est rempli ici
		...
		
		int valeur;
		MPI_Scatter(matrice, 1, MPI_INT, &valeur, 1, MPI_INT, 0, MPI_COMM_WORLD);
		printf("%d\n", valeur);

	} else {
		int valeur;
		MPI_Recv(&valeur, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%d\n", valeur);
	}
	MPI_Finalize();
	Return 0;
}