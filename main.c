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

void QueryOneItem(NeuralNetwork *nn)
{
    freopen("data.csv", "r", stdin);
    int correct = 0;
    Matrix v = {0, 0, NULL}, w = {0, 0, NULL};
    Fill(&v, 1, 784, 0);
    Fill(&w, 10, 1, 0);
    printf("Start to query one item:\n");
    int target = read(&v);
    Normalize(&v);
    QueryNN(nn, v, &w);
    Normalize(&w);
    int max_index = 0;
    printf("Possibilities:\n");
    for(int j = 0; j < 10; ++j)
        printf("%d     ", j);
    puts("");
    for(int j = 0; j < 10; ++j)
    {
        printf("%.03f ", w.data[j * w.col_size]);
        if(w.data[j * w.col_size] >= w.data[max_index * w.col_size])
            max_index = j;
    }
    freeMatrix(&v);
    freeMatrix(&w);
    printf("\nPrediction = %d\n", max_index);
}

int main()
{
    clock_t start, end;
    start = clock();
    printf("MNIST MLP Classifier, optimized version.\n");
    float rate = 0.10;
    printf("rate = %f\n\n", rate);
    srand(time(NULL));
    NeuralNetwork nn = InitNN(784, 200, 10, rate);
    printf("`nn` initialized.\n");
    Read(&nn, 60000);
    start = clock();
    for(int i = 0; i < 5; i++, rate *= 0.95, printf("Generation %d\n", i))
        Train(&nn, 60000);
    puts("Training finished.");
    ReadAndQuery(&nn, 10000);
    QueryOneItem(&nn);
    freeNN(&nn);
    end = clock();
    double elapsed_secs = (end - start) * 1.0 / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds.\n", elapsed_secs);   
    return 0;
}