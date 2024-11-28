#define _GNU_SOURCE
#define MULTI_THREAD_ENABLED 0
#include <sched.h>
#include <pthread.h>
#include "neuralnetwork.h"

NeuralNetwork InitNN(int innodes, int hidenodes, int outnodes, real learnrate)
{
    NeuralNetwork nn;
    nn.innodes = innodes;
    nn.hidenodes = hidenodes;
    nn.outnodes = outnodes;
    nn.learnrate = learnrate;
    nn.Weight_in_to_hidden = XavierRand(hidenodes, innodes);
    nn.Weight_hidden_to_out = XavierRand(outnodes, hidenodes);
    //nn.Weight_in_to_hidden = RandMatrixByRand(hidenodes, innodes);
    //nn.Weight_hidden_to_out = RandMatrixByRand(outnodes, hidenodes);
    return nn;
}

void TrainNN(NeuralNetwork *nn, Matrix input, Matrix target, int n)
{
    Matrix hidden = {0, 0, NULL}, output = {0, 0, NULL}, output_errors = {0, 0, NULL}, hidden_errors = {0, 0, NULL};
    Matrix inputs = {0, 0, NULL};
    inputs = Tr(input);
    Matrix targets = {0, 0, NULL};
    targets = Tr(target);
    Mul(nn->Weight_in_to_hidden, inputs, &hidden);
    Sigmoid(&hidden);
    Mul(nn->Weight_hidden_to_out, hidden, &output);
    Sigmoid(&output);
    Minus(targets, output, &output_errors);
    Matrix tr = {0, 0, NULL};
    tr = Tr(nn->Weight_hidden_to_out);
    Mul(tr, output_errors, &hidden_errors);
    #if MULTI_THREAD_ENABLED
    #warning "Warning: Multithread support may not be comprehensive."
    ErrorFeedbackCorrectionArgs arg1 = {&nn->Weight_hidden_to_out, output_errors, output, hidden, nn->learnrate};
    ErrorFeedbackCorrectionArgs arg2 = {&nn->Weight_in_to_hidden, hidden_errors, hidden, inputs, nn->learnrate};
    pthread_t thread1, thread2;
    cpu_set_t cpuset1, cpuset2;
    CPU_ZERO(&cpuset1);
    CPU_ZERO(&cpuset2);
    CPU_SET(n % 2, &cpuset1);
    CPU_SET((n) % 2, &cpuset2);
    pthread_create(&thread1, NULL, (void*)ErrorFeedbackCorrectionThread, &arg1);
    pthread_create(&thread2, NULL, (void*)ErrorFeedbackCorrectionThread, &arg2);
    pthread_setaffinity_np(thread1, sizeof(cpu_set_t), &cpuset1);
    pthread_setaffinity_np(thread2, sizeof(cpu_set_t), &cpuset2);
    #endif
    //ErrorFeedbackCorrectionThread(&arg1);
    //ErrorFeedbackCorrectionThread(&arg2);
    #if MULTI_THREAD_ENABLED
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    #endif
    #if !(MULTI_THREAD_ENABLED)
    ErrorFeedbackCorrection(&nn->Weight_hidden_to_out, output_errors, output, hidden, nn->learnrate);
    ErrorFeedbackCorrection(&nn->Weight_in_to_hidden, hidden_errors, hidden, inputs, nn->learnrate);
    #endif
    freeMatrix(&hidden);
    freeMatrix(&output);
    freeMatrix(&inputs);
    freeMatrix(&targets);
    freeMatrix(&tr);
    freeMatrix(&output_errors);
    freeMatrix(&hidden_errors);
}

void QueryNN(NeuralNetwork *nn, Matrix input, Matrix *output)
{
    Matrix hidden = {0, 0, NULL};
    Matrix inputs = {0, 0, NULL};
    inputs = Tr(input);
    Mul(nn->Weight_in_to_hidden, inputs, &hidden);
    Sigmoid(&hidden);
    Mul(nn->Weight_hidden_to_out, hidden, output);
    Sigmoid(output);
    freeMatrix(&hidden);
    freeMatrix(&inputs);
}

void freeNN(NeuralNetwork *nn)
{
    freeMatrix(&nn->Weight_in_to_hidden);
    freeMatrix(&nn->Weight_hidden_to_out);
}