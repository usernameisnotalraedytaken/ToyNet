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

void TrainNN(NeuralNetwork *nn, Matrix input, Matrix target)
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
    ErrorFeedbackCorrection(&nn->Weight_hidden_to_out, output_errors, output, hidden, nn->learnrate);
    ErrorFeedbackCorrection(&nn->Weight_in_to_hidden, hidden_errors, hidden, inputs, nn->learnrate);
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