#include <stdio.h>
#include <time.h>

int main(){
    asm volatile("xorl %%eax,%%eax\n cpuid \n" ::: "%eax", "%ebx", "%ecx", "%edx");
}