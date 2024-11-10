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
        vec->data[0][i] = u;
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
            w.data[0][j] = j == n ? 0.99 : 0.01;
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
        TrainNN(nn, TrainData[i], TrainTarget[i]);
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
            if(w.data[j][0] >= w.data[max_index][0])
                max_index = j;
        correct += max_index == target ? 1 : 0;
        freeMatrix(&v);
        freeMatrix(&w);
    }
    printf("Correct = %d/%d\n", correct, len);
}

void OutputWeights(NeuralNetwork *nn)
{
    freopen("weights1.txt", "w", stderr);
    int rows = nn->Weight_in_to_hidden.row_size;
    int cols = nn->Weight_in_to_hidden.col_size;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            fprintf(stderr, "%f\n", nn->Weight_in_to_hidden.data[i][j]);
    freopen("weights2.txt", "w", stderr);
    rows = nn->Weight_hidden_to_out.row_size;
    cols = nn->Weight_hidden_to_out.col_size;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            fprintf(stderr, "%f\n", nn->Weight_hidden_to_out.data[i][j]);
    printf("Weights outputted to files.\n");
    fclose(stderr);
}

int main()
{
    clock_t start, end;
    start = clock();
    printf("MNIST MLP Classifier Training Session, optimized version.\n");
    float rate = 0.10;
    printf("rate = %f\n\n", rate);
    srand(time(NULL));
    NeuralNetwork nn = InitNN(784, 256, 10, rate);
    printf("`nn` initialized.\n");
    printf("Network size: %d input, %d hidden, %d output.\n", nn.innodes, nn.hidenodes, nn.outnodes);
    Read(&nn, 60000);
    start = clock();
    for(int i = 0; i < 1; i++, rate *= 0.9, printf("Generation %d\n", i))
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
