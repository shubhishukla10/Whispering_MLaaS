#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>


unsigned int timestamp(void) {
          unsigned int bottom;
          unsigned int top;
          asm volatile("xorl %%eax,%%eax\n cpuid \n" ::: "%eax", "%ebx", "%ecx", "%edx"); // flush pipeline
          asm volatile("rdtsc\n" : "=a" (bottom), "=d" (top) );                                                      // read rdtsc
          asm volatile("xorl %%eax,%%eax\n cpuid \n" ::: "%eax", "%ebx", "%ecx", "%edx"); // flush pipeline again
          return bottom;
}

int main() {
    pid_t child1, child2, child3, child4, child5, child6, child7;
    int c=0, img=0, i=0;
    unsigned int time1, time2;
    char victim_command[100], spy_command[100], csv_name[100], dir_name[100];
    int time_all[1000];
    FILE *fpt;

    
    for(c=0; c<10; c++){
        printf("Class %d\n", c);
        sprintf(dir_name, "mkdir -p Attack_Timing_Data/Class%d", c);
        system(dir_name);
        for(img=0; img<10; img++){
            printf("Image %d\n", img);
            for(i=0; i<100; i++){
                sprintf(victim_command, "taskset -c 0 python3 Inference_victim_time_MLP_attack.py %d %d",c,img);
                if (!(child1 = fork())) {
                    // first child --> Victim Process
                    system(victim_command);
                    exit(0);
                } else if (!(child2 = fork())) {
                    // second child --> Other User 1 process
                    system("taskset -c 0 python3 Other_user1_inference.py");
                    // sleep(5);
                    exit(0);
                } else if (!(child3 = fork())) {
                    // third child --> Other user 2 process
                    system("taskset -c 0 python3 Other_user2_inference.py");
                    exit(0);
                } else if (!(child4 = fork())) {
                    // fourth child --> Other user 3 process
                    system("taskset -c 0 python3 Other_user3_inference.py");
                    exit(0);
                } else if (!(child5 = fork())) {
                    // fifth child --> Other user 4 process
                    system("taskset -c 0 python3 Other_user4_inference.py");
                    exit(0);
                } else if (!(child5 = fork())) {
                    // sixth child --> Other user 5 process
                    system("taskset -c 0 python3 Other_user5_inference.py");
                    exit(0);
                } else if (!(child7 = fork())) {
                    // seventh child --> Other user 6 process
                    system("taskset -c 0 python3 Other_user6_inference.py");
                    exit(0);
                } else {
                    // parent --> Spy process
                    sprintf(victim_command, "taskset -c 0 python3 Inference_spy_time_MLP_attack.py %d %d",c,img);
                    time1 = timestamp();
                    system(victim_command);
                    wait(&child1);
                    wait(&child2);
                    wait(&child3);
                    wait(&child4);
                    wait(&child5);
                    wait(&child6);
                    wait(&child7);
                    time2 = timestamp();
                    time_all[i] = time2 - time1;
                    sleep(5);
                }
            }

            sprintf(csv_name, "Attack_Timing_Data/Class%d/Overall_Inference_Time%d_Image%d.csv", c, c, img);
            fpt = fopen(csv_name, "w+");
            fprintf(fpt,",Time\n");
            for(int j=0; j<1000; j++){
                fprintf(fpt,"%d,%d\n", j, time_all[j]);
            }
            fclose(fpt);
        }
        
    }

    return 0;
}
