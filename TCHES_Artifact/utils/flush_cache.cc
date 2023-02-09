#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/utsname.h>

#define CACHE_SIZE 8000000

void flush(void *p) { asm volatile("clflush 0(%0)\n" : : "c"(p) : "rax"); }

void LLC_flush(){
    int i,j, a[CACHE_SIZE];
    for(i=0;i<CACHE_SIZE;i++){
        a[i]=i;
    }
    for(i=0;i<CACHE_SIZE;i++){
        j=a[i];
    }
}

int main(){
    LLC_flush();
}
