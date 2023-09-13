/* File:    omp_trap3.c
 * Purpose: Estimate definite integral (or area under curve) using the
 *          trapezoidal rule.  This version uses a parallel for directive
 *
 * Notes:   
 *   1.  The function f(x) is hardwired.
 *   2.  In this version, it's not necessary for n to be
 *       evenly divisible by thread_count.
 *
 * From Introduction to Parallel Programming (IPP):  Section 5.5 (pp. 224 and ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double f(double x);    // Function we're integrating
double Trap(double a, double b, int n, int thread_count); /

int main(void) {
   double  global_result = 0.0;
   double  a, b;
   int     n;
   int     thread_count;

   // Inputs. A better program would ask or receive those as main args.
   a = 0;
   b = 100;
   n = 10000;
   thread_count = 8;

   global_result = Trap(a, b, n, thread_count);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n", a, b, global_result);
   return 0;
}

/*
 * Function to be integraded
 */
double f(double x) {
   double return_val;
   return_val = x*x;
   return return_val;
}

/*
 *  Use trapezoidal rule to estimate definite integral
 */
double Trap(double a, double b, int n, int thread_count) {
   double  h, approx;

   h = (b-a)/n; 
   approx = (f(a) + f(b))/2.0; 
   #pragma omp parallel for \
    num_threads(thread_count) \
      reduction(+: approx)
   for (int i = 1; i <= n-1; i++){
     approx += f(a + i*h);
   }
   approx = h*approx; 

   return approx;
}  /* Trap */
