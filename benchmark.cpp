#include <cstdio>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

int main(int argc, char* argv[])
{
    puts("The program is a convient gadget to find a set of arguments for the best performance of the neural network.");
    puts("Now enter the range of hidden layers\' size:");
    int hid_st, hid_ed, step;
    printf("Enter the starting size: ");
    scanf("%d", &hid_st);
    printf("Enter the ending size: ");
    scanf("%d", &hid_ed);
    printf("Enter the step: ");
    scanf("%d", &step);
    puts("Now enter the range of learning rate:");
    float lr_st, lr_ed, lr_step;
    printf("Enter the starting learning rate: ");
    scanf("%f", &lr_st);
    printf("Enter the ending learning rate: ");
    scanf("%f", &lr_ed);
    printf("Enter the step: ");
    scanf("%f", &lr_step);
    for(int i = hid_st; i <= hid_ed; i += step)
    {
        for(float j = lr_st; j <= lr_ed; j += lr_step)
        {
            std::string cmd = "./toyNet_train.exe " + std::to_string(i) + " " + std::to_string(j) + ">> result.log";
            printf("Now running the command: %s\n", cmd.c_str());
            system(cmd.c_str());
        }
    }
    return 0;
}