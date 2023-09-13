#include <stdio.h>
#include <time.h>

#define MAXK 1e6 /*    被测函数最大重复调用次数    */

void hello();
int main(){
    int i;
    clock_t start, stop;
    double duration;
    start = clock();
    for(i=0; i<MAXK; i++){
        hello();
    }
    stop =  clock();
    long duacounts = stop - start;
    duration = ((double)(duacounts))/CLOCKS_PER_SEC/MAXK;
    //start==end, 10000, 1e5
    //start==end, 60000, 1e6
    printf("start = %ld, stop = %ld, duacounts = %d, duration = %4.12f\n", start, stop, duacounts, duration);//0.00003s左右
    return 0;
}

void hello(){
    printf("Hello world!");
}
