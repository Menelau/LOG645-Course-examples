#include <stdio.h>  


void foo( int a ){
  a *= 2;
}
int main(){
  printf("Example pass par value");
  int a = 12;
  printf( "Before foo: %d\n", a );
  foo( a );
  printf( "After foo: %d\n", a );
  return 0;
}
