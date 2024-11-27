#define _GNU_SOURCE // pour asprintf
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define TAILLE 34 
#define CHALEUR_INIT   180
#define CHALEUR_SOURCE 2800
//#define X_SOURCE TAILLE/2
//#define Y_SOURCE TAILLE/2
#define X_SOURCE 2
#define Y_SOURCE 1
#define NB_ITER 18
#define CX  .1
#define CY  .1

#define TAG_LIGNE 1
#define TAG_COLONNE 2

#define SWAP(a,b,type) {type ttttttttt=a;a=b;b=ttttttttt;}

void ecrireMatrice( double*, int, int, int );

int main( int argc, char** argv ){

    int N = TAILLE, k;
    int localN, largeur_grille;
    int rank, size, col_rank, col_size, row_rank, row_size;
    int ndims = 2;
    MPI_Comm cart_2D, row_comm, col_comm;
    int dims[ndims], periods[ndims], remain_dims[ndims], coord[ndims];
    MPI_Datatype type_lignes, type_colonnes;
    MPI_Request req[8];
    MPI_Status stat[8];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    largeur_grille = (int)sqrt(size);

    if(largeur_grille*largeur_grille != size){
        printf("Le nombre de processus doit etre un carre : size=P*P\n");
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    
    if(argc > 1) {
        N = atoi(argv[1]);
    }
    localN = N / largeur_grille;

    /* On alloue ce qu'on va avoir en local */
    double* surface = (double*) malloc( (localN+2)*(localN+2)*sizeof( double ) );
    double* tmp =  (double*) malloc( (localN+2)*(localN+2)*sizeof( double ) );

    for(int i = 0; i < (localN+2)*(localN+2); i++){
        surface[i] = CHALEUR_INIT;
    }
    
    /* Il y a un processus qui a une source chaude. On va la mettre vers le milieu
       mais on n'oublie pas qu'on a une ghost region dans le calcul de son indice. */
    int milieu = largeur_grille + (int)(largeur_grille/2);
    if(rank == milieu){
        printf("local N : %d\n", localN);
        printf("Source sur %d\n", rank);
        surface[(1+Y_SOURCE)*(localN+2) + 1+X_SOURCE] = CHALEUR_SOURCE;
    }

    /* On a besoin d'initialiser les bords de la matrice intermediaire.
       Attention, les indices vont de 0 a localN+1. */
    for(int j = 0; j < localN+2; j++) tmp[j] = CHALEUR_INIT;
    for(int j = 0; j < localN+2; j++) tmp[(localN+1)*(localN+2)+j] = CHALEUR_INIT;
    for(int i = 1; i < localN+1; i++) tmp[i*(localN+2)] = CHALEUR_INIT;
    for(int i = 1; i < localN+1; i++) tmp[i*(localN+2)+localN+1] = CHALEUR_INIT;

    /* On va communiquer sur une grille de processus 2D, donc on commence par 
       creer une topologie cartesienne 2D et des communicateurs dessus */
    
    dims[0] = largeur_grille;
    dims[1] = largeur_grille;
    periods[0] = 0;
    periods[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_2D);

    if(MPI_COMM_NULL == cart_2D) {
        fprintf(stderr, "Erreur pendant la creation de la topologie cartesienne\n");
        MPI_Comm_free(&cart_2D);
        free(surface);
        free(tmp);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    remain_dims[0] = 1;
    remain_dims[1] = 0;

    MPI_Cart_sub(cart_2D, remain_dims, &row_comm);

    if(MPI_COMM_NULL == row_comm) {
        fprintf(stderr, "Erreur pendant la creation des communicateurs de rangees");
        MPI_Comm_free(&cart_2D);
        MPI_Comm_free(&row_comm);
        free(surface);
        free(tmp);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    remain_dims[0] = 0;
    remain_dims[1] = 1;

    MPI_Cart_sub(cart_2D, remain_dims, &col_comm);

    if(MPI_COMM_NULL == row_comm) {
        fprintf(stderr, "Erreur pendant la creation des communicateurs de colonnes");
        MPI_Comm_free(&cart_2D);
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
        free(surface);
        free(tmp);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    /* On trouve qui on est sur ces communicateurs */
    
    MPI_Comm_size(col_comm, &col_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(row_comm, &row_rank);

    /* Creation des dataypes pour les lignes et les colonnes */
    
    MPI_Type_contiguous(localN+2, MPI_DOUBLE, &type_lignes);
    MPI_Type_commit(&type_lignes);

    MPI_Type_vector(localN+2, 1, localN+2, MPI_DOUBLE, &type_colonnes);
    MPI_Type_commit(&type_colonnes);
    
    /* Passons maintenant au calcul */

    for(int step = 0; step < NB_ITER; step++){

        /* On calcule l'interieur de notre sous-domaine */
        
        for(int i = 1; i < localN+1; i++){
           for(int j = 1; j < localN+1; j++){

               tmp[i*(localN+2) + j] = surface[i*(localN+2) + j] +
                   CX *( surface[(i+1)*(localN+2)+j] + surface[(i-1)*(localN+2)+j]
                         - 2 * surface[i*(localN+2) + j])
                   + CY * ( surface[i*(localN+2)+j+1] + surface[i*(localN+2)+j-1]
                            - 2 * surface[i*(localN+2) + j]);
           }
        }

        /* Echange avec les voisins */
        k = 0;
        if(col_rank != col_size - 1){ /* Est-ce que j'ai un voisin du dessus ? */
            MPI_Irecv(tmp, 1, type_lignes, col_rank + 1, TAG_LIGNE, col_comm, req+k);
            k++;
            MPI_Isend(tmp + (localN+2), 1, type_lignes, col_rank + 1, TAG_LIGNE, col_comm, req+k);
            k++;
        }
        
        if(col_rank != 0){ /* Est-ce que j'ai un voisin du dessous ? */
            MPI_Irecv(tmp + (localN+1) * (localN+2), 1, type_lignes, col_rank - 1, TAG_LIGNE, col_comm, req+k);
            k++;
            MPI_Isend(tmp + localN * (localN+2), 1, type_lignes, col_rank - 1, TAG_LIGNE, col_comm, req+k);
            k++;
        }

        if(row_rank != 0){ /* Est-ce que j'ai un voisin de gauche ? */
            MPI_Irecv(tmp, 1, type_colonnes, row_rank - 1, TAG_COLONNE, row_comm, req+k);
            k++;
            MPI_Isend(tmp + 1, 1, type_colonnes, row_rank - 1, TAG_COLONNE, row_comm, req+k);
            k++;
        }
        if(row_rank != row_size - 1){ /* Est-ce que j'ai un voisin du droite ? */
            MPI_Irecv(tmp + localN + 1, 1, type_colonnes, row_rank + 1, TAG_COLONNE, row_comm, req+k);
            k++;
            MPI_Isend(tmp + localN, 1, type_colonnes, row_rank + 1, TAG_COLONNE, row_comm, req+k);
            k++;
        }

        /* On attend que toutes les comm se fassent */
        MPI_Waitall(k, req, stat);

        /* On a fini l'iteration, on prepare la suivante */
        SWAP(surface, tmp, double*);
    }
    
    /* On ecrit le resultat dans un fichier */
    ecrireMatrice(surface, localN+2, localN+2, rank);

    /* On libere tout */
    free(surface);
    free(tmp);
    
    MPI_Comm_free(&cart_2D);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    
    MPI_Type_free(&type_lignes);
    MPI_Type_free(&type_colonnes);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void ecrireMatrice(double* mat, int lignes, int colonnes, int rank){
    FILE* fd;
    char* fname;
    asprintf( &fname, "chaleur_%d", rank );
    fd = fopen( fname, "w+" );

    /* On n'ecrit pas les ghost regions */
    for(int i = 1; i < lignes-1; i++){
        for(int j = 1; j < colonnes-1; j++){
            fprintf(fd, "%.2lf, ", mat[ i*colonnes + j ]);
        }
        fprintf(fd, "\n");
    }
    
    fclose(fd);
    free(fname);
}

