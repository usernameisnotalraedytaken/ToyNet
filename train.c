#include <stdio.h>
#include <time.h>
#include "neuralnetwork.h"

int read(Matrix *vec)
{
    int n = 0;
    char c1 = getchar();
    while(c1>'9'||c1<'0')c1=getchar();
    while(c1>='0'&&c1<='9')n=n*10+c1-'0',c1=getchar();
    vec->row_size = 1;
    vec->col_size = 28*28;
    for(unsigned i = 0; i < 28*28; ++i)
    {
        unsigned char u = 0;
        char c = getchar();
        while(c>'9'||c<'0')c=getchar();
        while(c>='0'&&c<='9')u=u*10+c-'0',c=getchar();
        vec->data[i] = u;
    }
    return n;
}

Matrix TrainData[60000], NoiseData[60000];
Matrix TrainTarget[60000];

void Read(NeuralNetwork *nn, int len)
{
    freopen("mnist_train.csv", "r", stdin);
    for(int i = 0; i < len; ++i)
    {
        Matrix v = {0, 0, NULL}, w = {0, 0, NULL};
        Fill(&v, 1, 784, 0);
        Fill(&w, 1, 10, 0);
        int n = read(&v);
        Normalize(&v);
        w.row_size = 1;
        w.col_size = 10;
        for(int j = 0; j < 10; ++j)
            w.data[j] = j == n ? 0.99 : 0.01;
        TrainData[i] = v;
        //NoiseData[i] = v;
        //addSpice(v, 0.05);
        TrainTarget[i] = w;
    }
}

void Train(NeuralNetwork *nn, int len)
{

    //for(int i = 0; i < len; ++i)
    //    TrainNN(nn, NoiseData[i], TrainTarget[i]);
    //nn->learnrate *= 0.97;
    for(int i = 0; i < len; ++i)
        TrainNN(nn, TrainData[i], TrainTarget[i], i);
}

void ReadAndQuery(NeuralNetwork *nn, int len)
{
    freopen("mnist_test.csv", "r", stdin);
    int correct = 0;
    for(int i = 0; i < len; ++i)
    {
        Matrix v = {0, 0, NULL}, w = {0, 0, NULL};
        Fill(&v, 1, 784, 0);
        Fill(&w, 10, 1, 0);
        int target = read(&v);
        Normalize(&v);
        QueryNN(nn, v, &w);
        int max_index = 0;
        for(int j = 0; j < 10; ++j)
            if(w.data[j * w.col_size] >= w.data[max_index * w.col_size])
                max_index = j;
        correct += max_index == target ? 1 : 0;
        freeMatrix(&v);
        freeMatrix(&w);
    }
    printf("Correct = %d/%d\n", correct, len);
}

void OutputWeights(NeuralNetwork *nn)
{
    FILE *fp = fopen("weights1.txt", "w");
    freopen("weights1.txt", "w", fp);
    int rows = nn->Weight_in_to_hidden.row_size;
    int cols = nn->Weight_in_to_hidden.col_size;
    fprintf(fp, "%d\n%d\n", rows, cols);
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            fprintf(fp, "%f\n", nn->Weight_in_to_hidden.data[i * cols + j]);
    fclose(fp);
    FILE *fp2 = fopen("weights2.txt", "w");
    freopen("weights2.txt", "w", fp2);
    rows = nn->Weight_hidden_to_out.row_size;
    cols = nn->Weight_hidden_to_out.col_size;
    fprintf(fp2, "%d\n%d\n", rows, cols);
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            fprintf(fp2, "%f\n", nn->Weight_hidden_to_out.data[i * cols + j]);
    printf("Weights successfully outputted to files.\n");
    fclose(fp2);
}

int main(int argc, char **argv)
{
    printf("Using C Standard : %ld\n", __STDC_VERSION__);
    clock_t start, end;
    start = clock();
    printf("MNIST MLP Classifier Training Session, optimized version.\n");
    float rate = 0.1;
    int round = 1;
    printf("rate = %f\n\n", rate);
    srand(time(NULL));
    NeuralNetwork nn;
    if(argc == 1)
    {
        LOG("No argument provided, using default values.");
        LOG("Initializing neural network with 784 input, 200 hidden, 10 output nodes, and learning rate of 0.1.");
        nn = InitNN(784, 200, 10, rate);
    }
    else if(argc == 3)
    {
        LOG("No learning rate provided, using default value of 0.1.");
        nn = InitNN(784, atoi(argv[1]), 10, rate);
    }
    else if(argc == 4)
    {
        nn = InitNN(784, atoi(argv[1]), 10, atof(argv[2]));
        round = atoi(argv[3]);
    }
    else if (argc > 4)
    {
        LOG("Too many arguments provided, using default values.");
        LOG("Initializing neural network with 784 input, 200 hidden, 10 output nodes, and learning rate of 0.1.");
        nn = InitNN(784, 200, 10, rate);
    }
    else
        nn = InitNN(784, atoi(argv[1]), 10, atof(argv[2]));
    printf("`nn` initialized.\n");
    printf("Network size: %d input, %d hidden, %d output.\n", nn.innodes, nn.hidenodes, nn.outnodes);
    Read(&nn, 60000);
    printf("Learning rate: %f\n", nn.learnrate);
    printf("Training for %d rounds.\n", round);
    start = clock();
    for(int i = 0; i < round; i++, rate *= 1, printf("Generation %d\n", i))
        Train(&nn, 60000);
    puts("Training finished.");
    ReadAndQuery(&nn, 10000);
    OutputWeights(&nn);
    freeNN(&nn);
    end = clock();
    double elapsed_secs = (end - start) * 1.0 / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds.\n", elapsed_secs);   
    return 0;
}