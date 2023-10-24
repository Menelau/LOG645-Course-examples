#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BORNEMIN -2.0
#define BORNEMAX  2.0
#define GRAIN    1e-8

double func( double value ){
    return 1.0/(value*value) + value*value + value * sin( value ) - 1.0/(value - 3.0);
}

void argmin( float gauche, float droite, double* arg, double* value ){
    double x, res, resmin, argm;
    x = gauche;
    res = func( x );
    resmin = res;
    argm = x;
    x += GRAIN;
    while( x <= droite ){
        res = func( x );
        if( res < resmin ){
            argm = x;
            resmin = res;
        }        
        x += GRAIN;
    }
    *arg = argm;
    *value = resmin;
}

int main(){
    int numthreads, tid;
    double droite, gauche, argm, valuemin, final_argmin, final_resmin;
    int entered = 0;
    double longueurintervalle;
    
#pragma omp parallel private( tid, droite, gauche, numthreads, argm, valuemin ) shared( final_argmin, final_resmin, entered )
    {
        numthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        longueurintervalle = (BORNEMAX - BORNEMIN) / numthreads;      
        gauche = BORNEMIN + tid * longueurintervalle;
        droite = BORNEMIN + ( tid + 1 ) * longueurintervalle;

        printf("Hello World from thread = %d / %d calcul de %.2lf à %.2lf\n", tid, numthreads, gauche, droite );

        argmin( gauche, droite, &argm, &valuemin );
#pragma omp critical ( nmthreads )
        {
            if( entered == 0 ){
                entered = 1;
                final_argmin = argm;
                final_resmin = valuemin;
            } else {
                if( valuemin < final_resmin ){
                    final_argmin = argm;
                    final_resmin = valuemin;
                }
            }
        }
    }
    printf( "Resultat : %.2lf\n", final_argmin );
    return EXIT_SUCCESS;
}
